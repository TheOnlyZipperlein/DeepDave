using DeepDave.Logging;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace DeepDave.Imaging {
    public class ImageLoader {
        public ImageLoader() {
        }

        /// <summary>
        /// Loads an image.
        /// </summary>
        /// <param name="pathToImage"></param>
        /// <param name="ideal"></param>
        /// <returns></returns>
        public ManagedImage LoadImage(String pathToImage, ManagedImage ideal = null) {
            return new ManagedImage(new Bitmap(pathToImage), ideal);
        }

        /// <summary>
        /// Loads all images in 
        /// </summary>
        /// <param name="pathToDir"></param>
        /// <param name="ideal"></param>
        /// <returns></returns>
        public List<ManagedImage> LoadImages(String pathToDir, ManagedImage ideal = null) {
            List<ManagedImage> images = new List<ManagedImage>();
            foreach(String file in Directory.GetFiles(pathToDir)) {
                if(file.EndsWith("png") | file.EndsWith("jpg")) {
                    images.Add(new ManagedImage(new Bitmap(file), ideal));
                }
            }
            if (images.Count == 0) Logger.WriteLine("ImageLoader: No images found in " + pathToDir);
            return images;
        }        
    }        
}
