using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Diagnostics;
using Emgu.CV.GPU;

namespace Detection
{
    
    public static class findObject
    {
       //=============================SVM Cassifier Data Training Tanaman=========================================
        private static float[] GetDataObjects(string objectData)
        {
            List<float> data = new List<float>();
            using (System.IO.StreamReader reader = new System.IO.StreamReader(objectData))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    var arr = new float[line.Length];
                    for (var i=0; i < line.Length; i++)
                    {
                        arr[i] = (float)Convert.ToDouble(line[i]);
                        data.Add(arr[i]);
                    }
                }
            }
            var array = data.ToArray();
            return array;
        }
        static Size blockSize = new Size(16, 16);
        static Size blockStride = new Size(8, 8);
        static Size winStride = new Size(8, 8);
        static Size cellSize = new Size(8, 8);
        static int nbins = 9;

        //=============================Feature Descriptor (HOG) Data Training Tanaman=============================
        public static Rectangle[] findObjects(Image<Bgr, Byte> image, out long processingTime, Size winSize, string dataFile)
        {
            Stopwatch watch;
            Rectangle[] regions;
            if(GpuInvoke.HasCuda)
            {
                using (GpuHOGDescriptor des = new GpuHOGDescriptor())
                {
                    des.SetSVMDetector(GpuHOGDescriptor.GetDefaultPeopleDetector());

                    watch = Stopwatch.StartNew();
                    using (GpuImage<Bgr, Byte> gpuImg = new GpuImage<Bgr,byte>(image))
                    using (GpuImage<Bgra,Byte> gpuBgra = gpuImg.Convert<Bgra,Byte>())
                    {
                        regions = des.DetectMultiScale(gpuBgra);
                    }
                }
            }
            else
            {
                using (HOGDescriptor des = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0.2, true)) 
                {
                    des.SetSVMDetector(GetDataObjects(dataFile));
                    watch = Stopwatch.StartNew();
                    regions = des.DetectMultiScale(image);
                }
            }
            watch.Stop();
            processingTime = watch.ElapsedMilliseconds;
            return regions;
        }
    }
}