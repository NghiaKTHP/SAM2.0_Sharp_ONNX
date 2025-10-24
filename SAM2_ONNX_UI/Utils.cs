using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAM2_ONNX
{
    public static class Utils
    {
        private static readonly Dictionary<int, Scalar> colors = new Dictionary<int, Scalar>
    {
        { 0, new Scalar(255, 0, 0) },     // Blue
        { 1, new Scalar(0, 255, 0) },     // Green  
        { 2, new Scalar(0, 0, 255) },     // Red
        { 3, new Scalar(255, 255, 0) },   // Cyan
        { 4, new Scalar(255, 0, 255) },   // Magenta
        { 5, new Scalar(0, 255, 255) },   // Yellow
        { 6, new Scalar(128, 0, 128) },   // Purple
        { 7, new Scalar(255, 165, 0) },   // Orange
        { 8, new Scalar(0, 128, 128) },   // Teal
        { 9, new Scalar(128, 128, 0) }    // Olive
    };

        public static Mat DrawMasks(Mat image, Dictionary<int, Mat> masks, double alpha = 0.5, bool drawBorder = true)
        {
            Mat maskImage = image.Clone();

            foreach (var kvp in masks)
            {
                int labelId = kvp.Key;
                Mat labelMask = kvp.Value;

                if (labelMask == null || labelMask.Empty())
                    continue;

                // Get color for this label, use default green if not found
                Scalar color = colors.ContainsKey(labelId) ? colors[labelId] : new Scalar(0, 255, 0);

                maskImage = DrawMask(maskImage, labelMask, color, alpha, drawBorder);
                break;
            }

            return maskImage;
        }

        public static Mat DrawMask(Mat image, Mat mask, Scalar color, double alpha = 0.5, bool drawBorder = true)
        {
            if (mask == null || mask.Empty())
                return image.Clone();

            Mat maskImage = image.Clone();

            // Ensure mask is the right size and type
            Mat processedMask = PreprocessMask(mask, image.Size());

            // Create a colored mask
            Mat coloredMask = Mat.Zeros(image.Size(), image.Type());

            // Apply threshold to ensure binary mask
            Mat binaryMask = new Mat();
            if (processedMask.Type() == MatType.CV_8UC1)
            {
                Cv2.Threshold(processedMask, binaryMask, 0, 255, ThresholdTypes.Binary);
            }
            else
            {
                // Convert to 8-bit if needed
                processedMask.ConvertTo(binaryMask, MatType.CV_8UC1, 255.0);
                Cv2.Threshold(binaryMask, binaryMask, 0, 255, ThresholdTypes.Binary);
            }

            // Set color where mask is active
            coloredMask.SetTo(color, binaryMask);

            // Blend the images
            Cv2.AddWeighted(image, 1 - alpha, coloredMask, alpha, 0, maskImage);

            if (drawBorder)
            {
                // Find contours and draw borders
                Mat[] contours;
                Mat hierarchy = new Mat();
                Cv2.FindContours(binaryMask, out contours, hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

                Cv2.DrawContours(maskImage, contours, -1, color, thickness: 2);

                // Dispose contours
                foreach (var contour in contours)
                    contour?.Dispose();
                hierarchy?.Dispose();
            }

            // Cleanup
            coloredMask?.Dispose();
            binaryMask?.Dispose();
            if (processedMask != mask)
                processedMask?.Dispose();

            return maskImage;
        }

        private static Mat PreprocessMask(Mat mask, Size targetSize)
        {
            Mat processedMask = mask;

            // Resize if size doesn't match
            if (mask.Size() != targetSize)
            {
                processedMask = new Mat();
                Cv2.Resize(mask, processedMask, targetSize, interpolation: InterpolationFlags.Nearest);
            }

            // Ensure single channel
            if (processedMask.Channels() > 1)
            {
                Mat singleChannel = new Mat();
                Cv2.CvtColor(processedMask, singleChannel, ColorConversionCodes.BGR2GRAY);
                if (processedMask != mask)
                    processedMask.Dispose();
                processedMask = singleChannel;
            }

            return processedMask;
        }

        // Alternative version that modifies the image in-place for better performance
        public static void DrawMaskInPlace(Mat image, Mat mask, Scalar color, double alpha = 0.5, bool drawBorder = true)
        {
            // Create a colored mask
            Mat coloredMask = Mat.Zeros(image.Size(), image.Type());
            coloredMask.SetTo(color, mask);

            // Create threshold mask
            Mat thresholdMask = new Mat();
            Cv2.Threshold(mask, thresholdMask, 0.01 * 255, 255, ThresholdTypes.Binary);

            // Apply blending only where mask is active
            Mat tempImage = image.Clone();
            Cv2.AddWeighted(tempImage, 1 - alpha, coloredMask, alpha, 0, coloredMask);
            coloredMask.CopyTo(image, thresholdMask);

            if (drawBorder)
            {
                // Find contours and draw borders
                Mat[] contours;
                Mat hierarchy = new Mat();
                Cv2.FindContours(mask, out contours, hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

                Cv2.DrawContours(image, contours, -1, color, thickness: 2);

                // Dispose contours
                foreach (var contour in contours)
                    contour?.Dispose();
                hierarchy?.Dispose();
            }

            // Cleanup
            coloredMask?.Dispose();
            thresholdMask?.Dispose();
            tempImage?.Dispose();
        }

        // Helper method to add more colors if needed
        public static void AddColor(int labelId, Scalar color)
        {
            colors[labelId] = color;
        }

        // Helper method to get color for a label
        public static Scalar GetColor(int labelId)
        {
            return colors.ContainsKey(labelId) ? colors[labelId] : new Scalar(0, 255, 0);
        }
    }
}
