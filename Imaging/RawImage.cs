using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace DeepDave.Imaging {
    public enum ColorFlag {
            red,
            blue,
            green
    }

    internal class RawImage {
        private Size size;
        private byte[,] red;
        private byte[,] green;
        private byte[,] blue;

        /// <summary>
        /// Creates a RawImage from the given values.
        /// </summary>
        /// <param name="red"></param>
        /// <param name="green"></param>
        /// <param name="blue"></param>
        /// <param name="size"></param>
        private RawImage(byte[,] red, byte[,] green, byte[,] blue, Size size) {
            this.red = red;
            this.green = green;
            this.blue = blue;
            this.size = size;
        }

        private RawImage(float[,] red, float[,] green, float[,] blue, Size size) {
            this.red = ConvertedArray(red);
            this.green = ConvertedArray(green);
            this.blue = ConvertedArray(blue);
            this.size = size;
        }

            /// <summary>
            /// Slow, but safe method to convert the RawImage to a bitmap.
            /// </summary>
            /// <returns></returns>
            internal Bitmap GetBitmapManaged() {
            Bitmap btmp = new Bitmap(size.Width, size.Height);
            for(int x=0; x < size.Width; x++) {
                for(int y = 0; y< size.Height; y++) {
                    btmp.SetPixel(x, y, Color.FromArgb(255, red[x, y], green[x, y], blue[x, y]));
                }
            }
            return btmp;
        }

        internal void SetColorArray(byte[,] colorArray, ColorFlag colorFlag) {
            switch (colorFlag) {
                case ColorFlag.red:
                    red= colorArray;
                    break;
                case ColorFlag.blue:
                    blue = colorArray;
                    break;
                case ColorFlag.green:
                    green = colorArray;
                    break;
            }
            throw new NotSupportedException("Not supported ColorFlag");
        }

        /// <summary>
        /// Creates a RawImage from the given Bitmap using Marshal.Copy.
        /// </summary>
        /// <param name="bmp"></param>
        /// <returns></returns>
        internal static RawImage FromBitmap(Bitmap bmp) {
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly,bmp.PixelFormat);
            
            IntPtr ptr = bmpData.Scan0;
            int bytesBuffer = bmpData.Stride * bmp.Height;
            byte[] unsortedRgbValues = new byte[bytesBuffer];
            System.Runtime.InteropServices.Marshal.Copy(ptr, unsortedRgbValues, 0, bytesBuffer); bmp.UnlockBits(bmpData);
            byte[,] red = new byte[bmp.Width, bmp.Height], green = new byte[bmp.Width, bmp.Height], blue = new byte[bmp.Width, bmp.Height];
            
            int x = 0, y = 0;
            for (int i = 0; i < unsortedRgbValues.Length; i++) {
                switch (i % 3) {
                    case 0:
                        red[x, y] = unsortedRgbValues[i];
                        break;
                    case 1:
                        green[x, y] = unsortedRgbValues[i];
                        break;
                    case 2:
                        blue[x, y] = unsortedRgbValues[i];
                        x++;
                        break;
                }
                if (x == bmp.Width) {
                    y++;
                    x = 0;
                }
            }

            return new RawImage(red, green, blue, rect.Size);
        }

        public static RawImage FromArray(float[][,] rgb) {
            var image = new RawImage(rgb[(int) ColorFlag.red], rgb[(int) ColorFlag.green], rgb[(int) ColorFlag.blue], 
                new Size(rgb[0].GetUpperBound(0)+1, rgb[0].GetUpperBound(1)+1));
            return image;
        }

        /// <summary>
        /// Return the requested color array of the Image.
        /// </summary>
        /// <param name="color"></param>
        /// <returns></returns>
        internal byte[,] GetColorArray(ColorFlag color) {
            switch(color) {
                case ColorFlag.red:
                    return red;
                case ColorFlag.blue:
                    return blue;
                case ColorFlag.green:
                    return green;
            }
            throw new NotSupportedException("Not supported ColorFlag");
        }

        private byte[,] ConvertedArray(float[,] array) {
            int xLimit = array.GetUpperBound(0)+1, yLimit = array.GetUpperBound(1)+1;
            var re = new byte[xLimit, yLimit];
            for(int x=0; x<xLimit; x++) {
                for(int y=0; y<yLimit; y++) {
                    re[x, y] = Convert.ToByte(array[x, y]);
                }
            }
            return re;
        }
    }
}

