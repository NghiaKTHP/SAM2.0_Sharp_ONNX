using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Numerics.Tensors;
using Point = OpenCvSharp.Point;
using MemoLibV2.OtherClasses;
using MemoLibV2.DeepLearning.BaseModel;
using MemoLibV2.ImageProcess;

namespace SAM2_ONNX
{
    public class SAM2 : Bindable, IDL_Model_2Phase
    {
        public string ModelName { get; set; }
        public string EncoderPath { get; set; }
        public string DecoderPath { get; set; }
        public string ClassesPath { get; set; }
        public eDevice Device { get; set; }

        private List<mmShape> _InputPositiveShape = new List<mmShape>();
        private List<mmShape> _InputNegativeShape = new List<mmShape>();
        public List<mmShape> InputPositiveShape { get => _InputPositiveShape; set { _InputPositiveShape = value; Notify(); } }

        public List<mmShape> InputNegativeShape { get => _InputNegativeShape; set { _InputNegativeShape = value; Notify(); } }


        public bool IsLoaded { get; set; } = false;

        #region DecoderProperty
        public OpenCvSharp.Size OriginImageSize { get => decoder.orig_im_size; private set { decoder.orig_im_size = value; Notify(); } }
        public OpenCvSharp.Size EncoderInputSize { get => decoder.encoder_input_size; private set { decoder.encoder_input_size = value; Notify(); } }
        public float MaskThreshold { get => decoder.mask_threshold; set { decoder.mask_threshold = value; Notify(); } }
        public int ScaleFactor { get => decoder.scale_factor; set { decoder.scale_factor = value; Notify(); } }
        public int MinMaskSize { get => decoder.minMaskSize; set { decoder.minMaskSize = value; Notify(); } }

        public int NumWorker { get; set; } = 0;

        #endregion

        public void Dispose()
        {
            this.UnloadModel();
            IsLoaded = false;
        }

        Sam2Encoder encoder = new Sam2Encoder();
        Sam2Decoder decoder = new Sam2Decoder();

        public void LoadModel(string encoderPath, string decoderPath, string classPath, eDevice device)
        {
            this.EncoderPath = encoderPath;
            this.DecoderPath = decoderPath;
            this.Device = device;
            this.LoadModel();
        }

        public void LoadModel()
        {
            this.encoder.LoadModel(this.EncoderPath, null, this.Device);
            this.decoder.LoadModel(this.DecoderPath, null, this.Device);

            this.EncoderInputSize = new OpenCvSharp.Size(encoder.input_shape[3], encoder.input_shape[2]);
            IsLoaded = true;
        }

        public List<mmIDL_Result> Predict(Mat mat)
        {
            (DenseTensor<float>, DenseTensor<float>, DenseTensor<float>) image_embeddings = encoder.Encoder(mat);
            this.OriginImageSize = mat.Size();

            (List<Mat> masks, float[] scores) = DecodeMask(InputPositiveShape, InputNegativeShape, image_embeddings);

            List<mmIDL_Result> results = new List<mmIDL_Result>();

            for (int i = 0; i < masks.Count; i++)
            {
                results.Add(new mmSamResult(masks[i], scores[i]));
            }

            return results;
        }

        private (List<Mat> mask, float[] scores) DecodeMask(List<mmShape> posShapes, List<mmShape> negShapes, (DenseTensor<float>, DenseTensor<float>, DenseTensor<float>) image_embeddings)
        {
            var (high_res_feats_0, high_res_feats_1, image_embed) = image_embeddings;

            var concat_coords = new List<mmPoint>();
            var concat_labels = new List<int>();

            foreach (var shape in posShapes)
            {
                if (shape is mmPoint)
                {
                    concat_coords.Add((mmPoint)shape);
                    concat_labels.Add(1);
                }
                if (shape is mmRectangle)
                {
                    concat_coords.AddRange(((mmRectangle)shape).ToListPoint());
                    concat_labels.AddRange(new int[] { 2, 3 });
                }
            }

            foreach (var shape in negShapes)
            {
                if (shape is mmPoint)
                {
                    concat_coords.Add((mmPoint)shape);
                    concat_labels.Add(0);
                }
            }

            return decoder.Decode(image_embed, high_res_feats_0, high_res_feats_1, concat_coords, concat_labels);
        }

        public void UnloadModel()
        {

        }


    }
}