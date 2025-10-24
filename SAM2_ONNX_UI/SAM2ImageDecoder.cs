using MemoLibV2.DeepLearning.BaseModel;
using MemoLibV2.ImageProcess;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAM2_ONNX
{
    internal class Sam2Decoder : OnnxModel
    {

        internal Size orig_im_size;
        internal Size encoder_input_size;
        internal float mask_threshold = 0.25f;
        internal int scale_factor = 4;
        internal int minMaskSize = 10000;

        public override void LoadModel()
        {
            var sessionOptions = new SessionOptions();

            if (Device == eDevice.CUDA)
            {
                sessionOptions.AppendExecutionProvider_CUDA(0);
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            }

            base.onnxSession = new InferenceSession(ModelPath, sessionOptions);
            var metaData = onnxSession.ModelMetadata;

            base.get_input_details();
            base.get_output_details();
        }

        public override void LoadModel(string modelPath, string classesPath, eDevice device)
        {
            this.ModelPath = modelPath;
            //this.ClassesPath = classesPath;
            this.Device = device;
            LoadModel();
        }


        public (List<Mat> masks, float[] scores) Decode(DenseTensor<float> image_embed, DenseTensor<float> high_res_feats_0,
        DenseTensor<float> high_res_feats_1, List<mmPoint> concat_coords, List<int> concat_labels)
        {
            return Predict(image_embed, high_res_feats_0, high_res_feats_1, concat_coords, concat_labels);
        }

        public (List<Mat> masks, float[] scores) Predict(DenseTensor<float> image_embed, DenseTensor<float> high_res_feats_0,
        DenseTensor<float> high_res_feats_1, List<mmPoint> concat_coords, List<int> concat_labels)
        {
            var inputs = PrepareInputs(image_embed, high_res_feats_0, high_res_feats_1, concat_coords, concat_labels);
            var outputs = Infer(inputs);
            return ProcessOutput(outputs);
        }

        private (List<Mat> masks, float[] scores) ProcessOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
        {
            var outputList = outputs.ToList();

            // Get tensors
            var maskTensor = outputList[0].AsTensor<float>();
            var scoreTensor = outputList[1].AsTensor<float>();

            // Extract scores (đã tối ưu)
            var scores = scoreTensor.ToArray();

            // Get dimensions
            var maskShape = maskTensor.Dimensions.ToArray();
            int numMasks = maskShape[1];
            int height = maskShape[2];
            int width = maskShape[3];
            int totalPixels = height * width;

            // OPTIMIZATION 1: Truy cập trực tiếp buffer thay vì ToArray()
            var maskData = maskTensor.ToArray(); // Giữ lại vì cần, nhưng tối ưu cách dùng

            var masks = new List<Mat>(numMasks); // Pre-allocate capacity

            // OPTIMIZATION 2: Tạo Mat trước và resize sau (tránh tạo Mat tạm)
            for (int maskIdx = 0; maskIdx < numMasks; maskIdx++)
            {
                int offset = maskIdx * totalPixels;

                // OPTIMIZATION 3: Tạo Mat trực tiếp với unsafe code
                Mat mask = new Mat(height, width, MatType.CV_8UC1);

                unsafe
                {
                    byte* maskPtr = (byte*)mask.DataPointer;

                    // OPTIMIZATION 4: Vectorization-friendly loop
                    Parallel.For(0, height, y =>
                    {
                        int rowStart = y * width;
                        byte* rowPtr = maskPtr + rowStart;

                        for (int x = 0; x < width; x++)
                        {
                            int idx = offset + rowStart + x;
                            rowPtr[x] = maskData[idx] > mask_threshold ? (byte)255 : (byte)0;
                        }
                    });
                }

                // OPTIMIZATION 5: Resize in-place nếu có thể
                Mat resizedMask = new Mat();
                Cv2.Resize(mask, resizedMask, orig_im_size, interpolation: InterpolationFlags.Linear);
                mask.Dispose();

                masks.Add(resizedMask);
            }

            outputs.Dispose();
            return (masks, scores);
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Infer(Dictionary<string, NamedOnnxValue> inputs)
        {
            var outputs = base.onnxSession.Run(inputs.Values);
            return outputs;
        }

        private unsafe (DenseTensor<float>, DenseTensor<float>) PreparePoints(List<mmPoint> point_coords, List<int> point_labels)
        {
            if (point_coords.Count == 0)
            {
                var empty_coords = new DenseTensor<float>(new int[] { 1, 0, 2 });
                var empty_labels = new DenseTensor<float>(new int[] { 1, 0 });
                return (empty_coords, empty_labels);
            }

            float x_scale = (float)encoder_input_size.Width / orig_im_size.Width;
            float y_scale = (float)encoder_input_size.Height / orig_im_size.Height;

            var input_point_coords = new DenseTensor<float>(new int[] { 1, point_coords.Count, 2 });
            var input_point_labels = new DenseTensor<float>(new int[] { 1, point_labels.Count });

            fixed (float* coords_ptr = input_point_coords.Buffer.Span)
            fixed (float* labels_ptr = input_point_labels.Buffer.Span)
            {
                for (int i = 0; i < point_coords.Count; i++)
                {
                    coords_ptr[i * 2] = (float)point_coords[i].X * x_scale;
                    coords_ptr[i * 2 + 1] = (float)point_coords[i].Y * y_scale;
                    labels_ptr[i] = point_labels[i];
                }
            }

            return (input_point_coords, input_point_labels);
        }

        private Dictionary<string, NamedOnnxValue> PrepareInputs(DenseTensor<float> image_embed, DenseTensor<float> high_res_feats_0,
       DenseTensor<float> high_res_feats_1, List<mmPoint> concat_coords, List<int> concat_labels)
        {
            var (input_point_coords, input_point_labels) = PreparePoints(concat_coords, concat_labels);

            int num_labels = input_point_labels.Dimensions[0];
            var mask_input = new DenseTensor<float>(new int[] {
            num_labels, 1,
            encoder_input_size.Height / scale_factor,
            encoder_input_size.Width / scale_factor
        });

            var has_mask_input = new DenseTensor<float>(new float[] { 0.0f }, new int[] { 1 });
            var original_size = new DenseTensor<int>(new int[] { orig_im_size.Height, orig_im_size.Width }, new int[] { 2 });

            var inputs = new Dictionary<string, NamedOnnxValue>
            {
                [input_names[0]] = NamedOnnxValue.CreateFromTensor(input_names[0], image_embed),
                [input_names[1]] = NamedOnnxValue.CreateFromTensor(input_names[1], high_res_feats_0),
                [input_names[2]] = NamedOnnxValue.CreateFromTensor(input_names[2], high_res_feats_1),
                [input_names[3]] = NamedOnnxValue.CreateFromTensor(input_names[3], input_point_coords),
                [input_names[4]] = NamedOnnxValue.CreateFromTensor(input_names[4], input_point_labels),
                [input_names[5]] = NamedOnnxValue.CreateFromTensor(input_names[5], mask_input),
                [input_names[6]] = NamedOnnxValue.CreateFromTensor(input_names[6], has_mask_input),
                //[input_names[7]] = NamedOnnxValue.CreateFromTensor(input_names[7], original_size)
            };

            return inputs;
        }


        public override List<mmIDL_Result> Predict(Mat mat)
        {
            throw new NotImplementedException();
        }

        protected override DenseTensor<float> PostProcess(Mat mat)
        {
            throw new NotImplementedException();
        }

        //internal override DenseTensor<float> PostProcess(Mat mat)
        //{
        //    throw new NotImplementedException();
        //}
    }
}
