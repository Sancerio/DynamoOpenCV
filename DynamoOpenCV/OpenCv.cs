using System;
using System.Collections.Generic;
using System.Drawing;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace DynamoOpenCV
{
    public class OpenCv
    {
        public static Mat ConvertBitMapToMat(Bitmap bitmap)
        {
            try
            {
                return OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);
            }
            catch (Exception e)
            {
                return null;
            }
        }

        public static Bitmap ConvertMatToBitMap(Mat mat)
        {
            try
            {
                return OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
            }
            catch (Exception e)
            {
                return null;
            }
        }

        public static Mat ConvertToGrayScale(Mat mat)
        {
            Mat grayMat = new Mat();
            Cv2.CvtColor(mat, grayMat, ColorConversion.RgbToGray);

            return grayMat;
        }

        public static Mat CreateHistogram(Mat grayScaleMat, int width = 260, int height = 200)
        {

            // Histogram view
            Mat render = new Mat(new OpenCvSharp.CPlusPlus.Size(width, height), MatType.CV_8UC3, Scalar.All(255));

            // Calculate histogram
            Mat hist = new Mat();
            int[] hdims = { 256 }; // Histogram size for each dimension
            Rangef[] ranges = { new Rangef(0, 256), }; // min/max 
            Cv2.CalcHist(
                new Mat[] { grayScaleMat },
                new int[] { 0 },
                null,
                hist,
                1,
                hdims,
                ranges);

            // Get the max value of histogram
            double minVal, maxVal;
            Cv2.MinMaxLoc(hist, out minVal, out maxVal);

            Scalar color = Scalar.All(100);
            // Scales and draws histogram
            hist = hist * (maxVal != 0 ? height / maxVal : 0.0);
            for (int j = 0; j < hdims[0]; ++j)
            {
                int binW = (int)((double)width / hdims[0]);
                render.Rectangle(
                    new OpenCvSharp.CPlusPlus.Point(j * binW, render.Rows),
                    new OpenCvSharp.CPlusPlus.Point((j + 1) * binW, render.Rows - (int)(hist.Get<float>(j))),
                    color,
                    -1);
            }

            return render;
        }

        public static void ShowMat(Mat mat, string title)
        {
            Cv2.ImShow(title, mat);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        public static List<Mat> PreviewCamera(bool savePictureMode)
        {
            // Opens a camera device
            var capture = new VideoCapture(0);
            if (!capture.IsOpened())
                return null;
            var returnedMat = new List<Mat>();

            var frame = new Mat();
            var returnedframe = new Mat();

            while (true)
            {
                // Read image
                capture.Read(frame);
                if (frame.Empty())
                    return null;

                Cv2.ImShow("Camera", frame);
                var key = Cv2.WaitKey(30);

                if (key == 27) //wait for esc key
                {
                    Cv2.DestroyAllWindows();
                    return returnedMat;
                }
                if (key == 115) //wait for s
                {
                    returnedframe = frame.Clone();
                    returnedMat.Add(returnedframe);
                }
            }
        }

        public static Mat GetMatFromList(List<Mat> list, int index)
        {
            return (index<list.Count)? list[index]:null;
        }

        public static Mat CannyEdge(Mat mat, int lowerThreshold = 50, int higherThreshold = 200)
        {
            Mat cannyMat = new Mat();
            Cv2.Canny(mat, cannyMat, lowerThreshold, higherThreshold);
            return cannyMat;
        }

        public static void Dft(string path)
        {
            Mat img = Cv2.ImRead(path, LoadMode.GrayScale);

            // expand input image to optimal size
            Mat padded = new Mat();
            int m = Cv2.GetOptimalDFTSize(img.Rows);
            int n = Cv2.GetOptimalDFTSize(img.Cols); // on the border add zero values
            Cv2.CopyMakeBorder(img, padded, 0, m - img.Rows, 0, n - img.Cols, BorderType.Constant, Scalar.All(0));

            // Add to the expanded another plane with zeros
            Mat paddedF32 = new Mat();
            padded.ConvertTo(paddedF32, MatType.CV_32F);
            Mat[] planes = { paddedF32, Mat.Zeros(padded.Size(), MatType.CV_32F) };
            Mat complex = new Mat();
            Cv2.Merge(planes, complex);

            // this way the result may fit in the source matrix
            Mat dft = new Mat();
            Cv2.Dft(complex, dft);

            // compute the magnitude and switch to logarithmic scale
            // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
            Mat[] dftPlanes;
            Cv2.Split(dft, out dftPlanes);  // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

            // planes[0] = magnitude
            Mat magnitude = new Mat();
            Cv2.Magnitude(dftPlanes[0], dftPlanes[1], magnitude);

            magnitude += Scalar.All(1);  // switch to logarithmic scale
            Cv2.Log(magnitude, magnitude);

            // crop the spectrum, if it has an odd number of rows or columns
            Mat spectrum = magnitude[
                new Rect(0, 0, magnitude.Cols & -2, magnitude.Rows & -2)];

            // rearrange the quadrants of Fourier image  so that the origin is at the image center
            int cx = spectrum.Cols / 2;
            int cy = spectrum.Rows / 2;

            Mat q0 = new Mat(spectrum, new Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
            Mat q1 = new Mat(spectrum, new Rect(cx, 0, cx, cy));  // Top-Right
            Mat q2 = new Mat(spectrum, new Rect(0, cy, cx, cy));  // Bottom-Left
            Mat q3 = new Mat(spectrum, new Rect(cx, cy, cx, cy)); // Bottom-Right

            // swap quadrants (Top-Left with Bottom-Right)
            Mat tmp = new Mat();
            q0.CopyTo(tmp);
            q3.CopyTo(q0);
            tmp.CopyTo(q3);

            // swap quadrant (Top-Right with Bottom-Left)
            q1.CopyTo(tmp);
            q2.CopyTo(q1);
            tmp.CopyTo(q2);

            // Transform the matrix with float values into a
            Cv2.Normalize(spectrum, spectrum, 0, 1, NormType.MinMax);

            // Show the result
            Cv2.ImShow("Spectrum Magnitude", spectrum);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        public static void Fast(string path)
        {
            using (Mat imgSrc = new Mat(path, LoadMode.Color))
            using (Mat imgGray = new Mat())
            using (Mat imgDst = imgSrc.Clone())
            {
                Cv2.CvtColor(imgSrc, imgGray, ColorConversion.BgrToGray, 0);

                KeyPoint[] keypoints;
                Cv2.FAST(imgGray, out keypoints, 50, true);

                foreach (KeyPoint kp in keypoints)
                {
                    imgDst.Circle(kp.Pt, 3, CvColor.Red, -1, LineType.AntiAlias, 0);
                }

                Cv2.ImShow("FAST", imgDst);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }
        }
    }
}
