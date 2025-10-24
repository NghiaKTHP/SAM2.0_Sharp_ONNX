using MemoLibV2.DeepLearning.BaseModel;
using MemoLibV2.DeepLearning.Tools;
using MemoLibV2.ImageProcess;
using MemoLibV2.ImageProcess.MemoDataType.CalculationDataType;
using MemoLibV2.ImageProcess.MemoTool.MemoImageProcess;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using Xceed.Wpf.Toolkit.PropertyGrid.Attributes;
using static MemoLibV2.PropertyGridCustomControl.btnLoadBitmap;

namespace SAM2_ONNX
{
    internal class Sam2DemoTool : mmImageProcessTool
    {
        [Category("Output Params")]
        [Description("")]
        [Xceed.Wpf.Toolkit.PropertyGrid.Attributes.ExpandableObject]
        public new DeepLearningOutputParam OutputParam { get; set; } = new DeepLearningOutputParam();

        private string _EncoderPath;
        private string _DecoderPath;
        private string _ClassPath;
        private eDevice _Device;


        [Category("Promt Shape")]
        [NewItemTypes(typeof(mmRectangle), typeof(mmPoint))]
        public List<mmShape> InputPositiveShape { get => Model.InputPositiveShape; set { Model.InputPositiveShape = value; Notify(); } }

        [Category("Promt Shape")]
        [NewItemTypes(typeof(mmPoint))]
        public List<mmShape> InputNegativeShape { get => Model.InputNegativeShape; set { Model.InputNegativeShape = value; Notify(); } }


        [Category("Run Params")]
        public eDevice Device { get => _Device; set { _Device = value; } }

        [Category("Run Params")]
        [Editor(typeof(btnEditCustomFullSize), typeof(btnEditCustomFullSize))]
        public string ReLoadModelButton { get; set; }

        [Category("Run Params")]
        public float MaskThreshold { get => _Model.MaskThreshold; set { _Model.MaskThreshold = value; Notify(); } }

        [Category("Run Params")]
        public int ScaleFactor { get => _Model.ScaleFactor; set { _Model.ScaleFactor = value; Notify(); } }

        [Category("Run Params")]
        public int MinMaskSize { get => _Model.MinMaskSize; set { _Model.MinMaskSize = value; Notify(); } }

        [Category("Run Params")]
        [Description("Path to encoder file")]
        [Editor(typeof(btnLoadDeepLearningModel), typeof(btnLoadDeepLearningModel))]
        public string EncoderPath { get => _EncoderPath; set { _EncoderPath = value; Notify(); } }

        [Category("Run Params")]
        [Description("Path to decoder file")]
        [Editor(typeof(btnLoadDeepLearningModel), typeof(btnLoadDeepLearningModel))]
        public string DecoderPath { get => _DecoderPath; set { _DecoderPath = value; Notify(); } }


        public void OnReLoadModelButtonEditRequested(object data)
        {
            CreateModel();
        }

        private SAM2 _Model = new SAM2();
        [Category("DeepLearning Model")]
        [Xceed.Wpf.Toolkit.PropertyGrid.Attributes.ExpandableObject]
        public SAM2 Model { get => _Model; set { _Model = value; Notify(); } }



        public void CreateModel()
        {
            if (this.Model == null) Model = new SAM2();

            this.Model?.UnloadModel();
            this.Model.LoadModel(this.EncoderPath, this.DecoderPath, "", Device);
        }

        public override void Run()
        {
            try
            {
                DateTime st = DateTime.Now;

                if (InputParam == null)
                {
                    RunState = ToolState.Error;
                    RunMessage = "InputParam is null!";
                    Ran?.Invoke(this, EventArgs.Empty);
                    return;
                }

                if (InputParam.InputImage == null)
                {
                    RunState = ToolState.Error;
                    RunMessage = "Input Image is null!";
                    Ran?.Invoke(this, EventArgs.Empty);
                    return;
                }


                if (this.Model == null || !this.Model.IsLoaded)
                {
                    this.CreateModel();
                }

                if (Model == null)
                {
                    RunState = ToolState.Error;
                    RunMessage = "Model not loaded!";
                    Ran?.Invoke(this, EventArgs.Empty);
                    return;
                }

                if (InputPositiveShape == null || InputPositiveShape.Count == 0)
                {
                    RunState = ToolState.Error;
                    RunMessage = "InputPositiveShape is Wrong";
                    Ran?.Invoke(this, EventArgs.Empty);
                    return;
                }

                this.ClearImageView();

                AddImageView(nameof(InputParam.InputImage), new mmImage(InputParam.InputImage));
                this.OutputParam.Dispose();


                List<mmIDL_Result> results = this.Model.Predict(this.InputParam.InputImage);

                List<mmShape> graphics = new List<mmShape>();
                if (this.IsGenerateOutputImage)
                {
                    foreach (mmShape shape in InputPositiveShape)
                    {
                        shape.Stroke = new SolidColorBrush(Colors.Blue);
                        //shape.StrokeThickness *= 5;
                        shape.Name = "Positive";
                        graphics.Add(shape);
                    }

                    foreach (mmShape shape in InputNegativeShape)
                    {

                        shape.Stroke = new SolidColorBrush(Colors.Red);
                        //shape.StrokeThickness *= 5;
                        shape.Name = "Negative";
                        graphics.Add(shape);


                    }
                }

                for (int i = 0; i < results.Count; i++)
                {
                    AddImageView($"Mask_{i + 1}", new mmImage((results[i] as mmSamResult).Mask, graphics));
                }

                this.OutputParam.Results = results;

                Mat mat = new Mat();
                this.InputParam.InputImage.CopyTo(mat);






                AddImageView(nameof(OutputParam.OutputImage), new mmImage(mat));

                DateTime endTime = DateTime.Now;
                TactTime = endTime - st;
                RunState = ToolState.Done;
                RunMessage = "Done";
                Ran?.Invoke(this, EventArgs.Empty);

                if (this.IsSaveOutputImage)
                {
                    this.OutputParam.OutputImage.Save(PathSaveOutputImage, endTime.ToString("yyyy_MM_dd_HH_mm_ss_fff"), SaveImageFormat);
                }
            }

            catch (Exception ex)
            {

            }



        }
    }
}
