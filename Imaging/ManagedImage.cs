using System;
using System.Drawing;

namespace DeepDave.Imaging {
    public class ManagedImage {
        private RawImage image, calculated;
        private ManagedImage myIdeal;

        /// <summary>
        /// Creates a new ManagedImage from a bitmap.
        /// </summary>
        /// <param name="btmp"></param>
        /// <param name="ideal">The managedImaged the calculated output should be compared to.</param>
        public ManagedImage(Bitmap btmp, ManagedImage ideal = null) {
            this.image = RawImage.FromBitmap(btmp);
            this.myIdeal = this;
            calculated = null;
        }

        public void SetNetworkOutput(float[][,] rgb) {
            calculated = RawImage.FromArray(rgb);
        }

        public NetworkInput GetNetworkInput() {
            var rgb = new byte[3][,];
            var rgbShould = new byte[3][,];
            rgb[(int)ColorFlag.blue] = image.GetColorArray(ColorFlag.blue);
            rgb[(int)ColorFlag.red] = image.GetColorArray(ColorFlag.red);
            rgb[(int)ColorFlag.green] = image.GetColorArray(ColorFlag.green);
            rgbShould[(int)ColorFlag.blue] = myIdeal.GetRawColorArray(ColorFlag.blue);
            rgbShould[(int)ColorFlag.red] = myIdeal.GetRawColorArray(ColorFlag.red);
            rgbShould[(int)ColorFlag.green] = myIdeal.GetRawColorArray(ColorFlag.green);
            return new NetworkInput(rgb, rgbShould);
        }

        /// <summary>
        /// Returns the given input array.
        /// </summary>
        /// <param name="flag"></param>
        /// <returns></returns>
        public byte[,] GetRawColorArray(ColorFlag flag) {
            return image.GetColorArray(flag);
        }

        /// <summary>
        /// Returns the calcuted output of the neural network.
        /// </summary>
        /// <param name="flag"></param>
        /// <returns></returns>
        public byte[,] GetCalculatedArray(ColorFlag flag) {
            if (calculated == null) throw new NullReferenceException("This was not yet calculated.");
            return calculated.GetColorArray(flag);
        }

        public void SetCalculatedArray(byte[,] colorArray, ColorFlag flag) {
            image.SetColorArray(colorArray, flag);
        }

        /// <summary>s
        /// Return the best case array.
        /// </summary>
        /// <param name="flag"></param>
        /// <returns></returns>
        public byte[,] GetIdealColorArray(ColorFlag flag) {
            if (myIdeal == null) return image.GetColorArray(flag);
            return myIdeal.GetRawColorArray(flag);
        }

        public Bitmap GetBitmap() {
            return image.GetBitmapManaged();
        }

        public Bitmap GetBitmapCalc() {
            return calculated.GetBitmapManaged();
        }
    }
}
