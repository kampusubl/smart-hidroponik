using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.GPU;

namespace Detection
{
    public partial class Form1 : Form
    {
        private Capture _capture = null;
        private bool _captureProgress;
        float X, Y;
        public Form1()
        {
            InitializeComponent();
            try
            {
                _capture = new Capture();
                _capture.ImageGrabbed += ProcessFrame;
            }
            catch (NullReferenceException excpt)
            {
                MessageBox.Show(excpt.Message);
            }
        }

        public void ProcessFrame(object sender, EventArgs arg)
        {
            Image<Bgr, Byte> frame = _capture.RetrieveBgrFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_LINEAR).Copy(); ;
            int detected_human = 0;
            long processingTime;
            MCvFont f = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, 0.7);

            List<Rectangle> dataBound = new List<Rectangle>();
            List<string> dataTrain = new List<string>();

            string fileSawi = "D:\\sawi.txt";
            Size winSizeSawi = new Size(64, 128);
            Rectangle[] resultCabinet = findObject.findObjects(frame, out processingTime, winSizeSawi, fileSawi);
            foreach (Rectangle rect in resultCabinet)
            {
                ++detected_human;
                frame.Draw(rect, new Bgr(Color.Red), 2);
                Point HOGLocation = rect.Location;
                X = HOGLocation.X;
                Y = HOGLocation.Y;

                frame.Draw("[" + detected_human + "]" + "botol", ref f, new Point(rect.Left, rect.Top - 5), new Bgr(Color.Yellow));
            }

            imageBox1.Image = frame;
            
        }

        private void btn_start_Click(object sender, EventArgs e)
        {
            if(_capture != null)
            {
                if(_captureProgress)
                {
                    btn_start.Text = "start";
                    _capture.Pause();
                }
                else
                {
                    btn_start.Text = "stop";
                    _capture.Start();
                }
                _captureProgress = !_captureProgress;
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
