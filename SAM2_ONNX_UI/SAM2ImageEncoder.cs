using MemoLibV2.DeepLearning.BaseModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAM2_ONNX
{
    internal class Sam2Encoder : OnnxModel
    {
        public int img_height;
        public int img_width;

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

        public (DenseTensor<float>, DenseTensor<float>, DenseTensor<float>) Encoder(Mat mat)
        {
            DenseTensor<float> input_tensor = PostProcess(mat);
            List<DenseTensor<float>> outputs = Infer(input_tensor);
            return Process_Output(outputs);
        }

        public (DenseTensor<float>, DenseTensor<float>, DenseTensor<float>) Process_Output(List<DenseTensor<float>> outputs)
        {
            return (outputs[0], outputs[1], outputs[2]);
        }

        private List<DenseTensor<float>> Infer(DenseTensor<float> input_tensor)
        {
            List<NamedOnnxValue> reuseableInputs = new List<NamedOnnxValue>(1);
            reuseableInputs.Add(NamedOnnxValue.CreateFromTensor(input_names[0], input_tensor));

            // Run inference
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = base.onnxSession.Run(reuseableInputs);
            reuseableInputs.Clear();

            List<DenseTensor<float>> reuseableOutputs = new List<DenseTensor<float>>();
            var enumerator = results.GetEnumerator();
            while (enumerator.MoveNext())
            {
                DisposableNamedOnnxValue result = enumerator.Current;
                if (result.AsTensor<float>() is DenseTensor<float> tensor)
                {
                    reuseableOutputs.Add(tensor);
                }
            }
            results.Dispose();
            enumerator.Dispose();
            return new List<DenseTensor<float>>(reuseableOutputs);
        }

        public override List<mmIDL_Result> Predict(Mat mat)
        {
            throw new NotImplementedException();
        }

        float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
        float[] std = new float[] { 0.229f, 0.224f, 0.225f };
        float inv_std0 = 1.0f / 0.229f;
        float inv_std1 = 1.0f / 0.224f;
        float inv_std2 = 1.0f / 0.225f;

        protected override DenseTensor<float> PostProcess(Mat image)
        {
            img_height = image.Height;
            img_width = image.Width;

            // 1. Resize trước khi convert color (thường nhanh hơn)
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(input_width, input_height),
                       interpolation: InterpolationFlags.Linear);

            // 2. Convert sang RGB và normalize trong một bước
            Mat processed_img = new Mat();
            Cv2.CvtColor(resized, processed_img, ColorConversionCodes.BGR2RGB);
            processed_img.ConvertTo(processed_img, MatType.CV_32FC3, 1.0 / 255.0);
            resized.Dispose();

            // 3. Sử dụng unsafe code để truy cập trực tiếp memory
            var input_tensor = new DenseTensor<float>(new[] { 1, 3, input_height, input_width });

            unsafe
            {
                float* imgPtr = (float*)processed_img.DataPointer;
                int channels = processed_img.Channels();

                Parallel.For(0, input_height, y =>
                {
                    int rowOffset = y * input_width * channels;

                    for (int x = 0; x < input_width; x++)
                    {
                        int pixelOffset = rowOffset + x * channels;

                        // Direct memory access - nhanh hơn Get<Vec3f>
                        input_tensor[0, 0, y, x] = (imgPtr[pixelOffset] - mean[0]) * inv_std0;
                        input_tensor[0, 1, y, x] = (imgPtr[pixelOffset + 1] - mean[1]) * inv_std1;
                        input_tensor[0, 2, y, x] = (imgPtr[pixelOffset + 2] - mean[2]) * inv_std2;
                    }
                });
            }

            processed_img.Dispose();
            return input_tensor;
        }
    }
}
