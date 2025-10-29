# SAM2 ONNX UI

A C# WPF application for running Meta's Segment Anything Model 2 (SAM2) using ONNX Runtime for efficient image segmentation. Also work with Sam 2.0 and Sam 2.1

## Overview

This project provides a Windows desktop application that implements SAM2 (Segment Anything Model 2) for interactive image segmentation. The application uses ONNX Runtime for inference, allowing you to run the model on both CPU and CUDA-enabled GPUs.

## Features

- **Interactive Segmentation**: Click points or draw rectangles to segment objects in images
- **Positive/Negative Prompts**: Support for both positive (include) and negative (exclude) point prompts
- **GPU Acceleration**: CUDA support for faster inference
- **Real-time Visualization**: View segmentation masks in real-time
- **Configurable Parameters**: Adjust mask threshold, scale factor, and minimum mask size
- **Two-Phase Architecture**: Separate encoder and decoder models for optimal performance

## Requirements

- .NET Framework 4.8.1
- Windows OS (x64)
- ONNX Runtime 1.22.1
- OpenCV Sharp 4.9.0
- CUDA-capable GPU (optional, for GPU acceleration)

## Dependencies

The project uses the following main dependencies:

- **Microsoft.ML.OnnxRuntime** (1.22.1) - ONNX model inference
- **OpenCvSharp4** (4.9.0) - Image processing and computer vision
- **System.Numerics.Tensors** (9.0.8) - Tensor operations
- **MemoLibV2** - Custom libraries for deep learning and image processing
- **Xceed.Wpf.Toolkit** - WPF UI controls

## Project Structure

```
SAM2_ONNX_UI/
├── SAM2.cs                  # Main SAM2 model wrapper
├── SAM2ImageEncoder.cs      # Image encoder implementation
├── SAM2ImageDecoder.cs      # Mask decoder implementation
├── Sam2DemoTool.cs          # UI tool for demonstration
├── MainWindow.xaml          # Main application window
├── MainWindow.xaml.cs       # Window code-behind
├── Utils.cs                 # Utility functions
└── Dll/                     # Required DLL dependencies
```

## Key Components

### SAM2 Model ([SAM2.cs](SAM2.cs))

The main model class that orchestrates the two-phase segmentation process:
- Loads encoder and decoder ONNX models
- Manages positive and negative shape prompts
- Coordinates image encoding and mask decoding
- Supports both CPU and CUDA devices

### Sam2Encoder ([SAM2ImageEncoder.cs](SAM2ImageEncoder.cs))

Handles image encoding with optimizations:
- Resizes input images to encoder dimensions
- Normalizes images using mean/std normalization
- Uses unsafe code and parallel processing for performance
- Outputs high-resolution features and image embeddings

### Sam2Decoder ([SAM2ImageDecoder.cs](SAM2ImageDecoder.cs))

Processes prompts and generates segmentation masks:
- Accepts point and bounding box prompts
- Generates masks with confidence scores
- Supports mask threshold and size filtering
- Optimized with parallel processing and unsafe code

## Usage

### Loading the Model

1. Launch the application
2. Set the paths to your SAM2 encoder and decoder ONNX models
3. Select the device (CPU or CUDA)
4. Click "ReLoad Model" button

### Performing Segmentation

1. Load an input image
2. Add positive prompts:
   - Add points on the object you want to segment
   - Add rectangles around the object
3. (Optional) Add negative prompts:
   - Add points on areas to exclude
4. Run the segmentation
5. View the generated masks

### Configuration Parameters

- **MaskThreshold** (default: 0.25): Threshold for binary mask generation
- **ScaleFactor** (default: 4): Scale factor for mask processing
- **MinMaskSize** (default: 10000): Minimum mask size in pixels
- **Device**: CPU or CUDA

## Model Files

You need to provide two ONNX model files:

1. **Encoder Model**: Processes the input image and generates embeddings
2. **Decoder Model**: Takes embeddings and prompts to generate segmentation masks

### Downloading Pre-converted ONNX Models

You can download pre-converted SAM2 ONNX models from Hugging Face:

**[https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/tree/main](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/tree/main)**

Available model variants:
- `sam2_hiera_tiny` - Smallest and fastest model
- `sam2_hiera_small` - Balance between speed and accuracy
- `sam2_hiera_base_plus` - Higher accuracy
- `sam2_hiera_large` - Best accuracy, slower inference

For each variant, download both:
- `image_encoder.onnx` - The encoder model
- `image_decoder.onnx` - The decoder model

### Example Model Setup

1. Visit the Hugging Face repository
2. Choose a model variant (e.g., `sam2_hiera_tiny`)
3. Download both `image_encoder.onnx` and `image_decoder.onnx`
4. Place them in a local folder
5. In the application, set the encoder and decoder paths to these files

## Code Example

```csharp
// Initialize SAM2 model
var sam2 = new SAM2();
sam2.LoadModel(encoderPath, decoderPath, "", eDevice.CUDA);

// Add positive point prompts
sam2.InputPositiveShape.Add(new mmPoint(100, 100));

// Add positive rectangle prompt
sam2.InputPositiveShape.Add(new mmRectangle(50, 50, 200, 200));

// Add negative point prompt
sam2.InputNegativeShape.Add(new mmPoint(150, 150));

// Run prediction
List<mmIDL_Result> results = sam2.Predict(inputImage);

// Get mask and score
var mask = (results[0] as mmSamResult).Mask;
var score = (results[0] as mmSamResult).Score;
```

## Performance Optimizations

The implementation includes several optimizations:

- **Unsafe Code**: Direct memory access for faster tensor operations
- **Parallel Processing**: Multi-threaded image processing
- **Pre-allocation**: Minimizes memory allocations during inference
- **Efficient Tensor Operations**: Optimized normalization and data conversion

## Building the Project

1. Clone the repository
2. Restore NuGet packages
3. Ensure all DLL dependencies are in the `Dll/` folder
4. Build using Visual Studio 2017 or later

```bash
msbuild SAM2_ONNX_UI.csproj /p:Configuration=Release
```

## License

This project uses Meta's Segment Anything Model 2 (SAM2), which is licensed under the **Apache License 2.0**.

### Model License
- **SAM2 Model**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- You are free to use, modify, and distribute the SAM2 models for both commercial and non-commercial purposes
- See the [official SAM2 repository](https://github.com/facebookresearch/segment-anything-2) for full license details

### Implementation Code
This implementation code is provided as-is and follows the respective licenses of its dependencies:
- **Microsoft.ML.OnnxRuntime**: MIT License
- **OpenCvSharp4**: Apache License 2.0
- **Other dependencies**: Please refer to their respective license files

## Acknowledgments

- Meta AI for the Segment Anything Model 2
- Microsoft for ONNX Runtime
- OpenCvSharp contributors

## Troubleshooting

### Model Loading Issues
- Ensure ONNX model files are valid and compatible with ONNX Runtime 1.22.1
- Check that the model paths are correct

### CUDA Issues
- Verify CUDA drivers are installed
- Ensure CUDA version is compatible with ONNX Runtime
- Try CPU device if CUDA fails

### Performance Issues
- Use CUDA device for better performance
- Reduce image resolution if necessary
- Adjust the number of worker threads

## Contact

For issues and questions, please refer to the project repository.
