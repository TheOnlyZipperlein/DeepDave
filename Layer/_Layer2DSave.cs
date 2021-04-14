using DeepDave.Helper;
using DeepDave.Helper.AbstractionClasses;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeepDave.Layer {
    public abstract partial class Layer2D : Saveable {
        /// <summary>
        /// Saves this Layer.
        /// </summary>
        /// <param name="writer"></param>
        void Saveable.Save(StreamWriter writer) {
            writer.WriteLine("Type: " + this.GetType().FullName);
            writer.WriteLine("Function: " + this.function);
            int sliceCount = this.bias.Length;
            writer.WriteLine("SliceCount: " + this.bias.Length);
            for (int i = 0; i < sliceCount; i++) {
                this.SaveBuffer(writer, variable[i], i, nameof(variable));
                this.SaveBuffer(writer, bias[i], i, nameof(bias));
                this.SaveBuffer(writer, derived[i], i, nameof(derived));
                this.SaveBuffer(writer, error[i], i, nameof(error));
                this.SaveBuffer(writer, activated[i], i, nameof(activated));
                this.SaveBuffer(writer, sumInput[i], i, nameof(sumInput));
                this.SaveBuffer(writer, weight[i], i, nameof(weight));
            }
        }

        /// <summary>
        /// Saves a 1DMemoryBuffer.
        /// </summary>
        protected void SaveBuffer(StreamWriter writer, MemoryBuffer<float> buffer, int sliceIndex, string name) {
            if (buffer.Equals(GPUHelper.dummyBuffer2D)) return;
            writer.WriteLine("buffer2D: " + name + " sliceIndex: " + sliceIndex);
            var arr = buffer.GetAsArray();
            var xBound = arr.GetUpperBound(0) + 1; 
            writer.WriteLine("xBound: " + xBound);
            for (int x = 0; x < xBound; x++) {
                    writer.Write(arr[x] + " ");
                
            }
        }
        /// <summary>
        /// Saves a 2DMemoryBuffer.
        /// </summary>
        protected void SaveBuffer(StreamWriter writer, MemoryBuffer2D<float> buffer, int sliceIndex, string name) {
            if (buffer.Equals(GPUHelper.dummyBuffer2D)) return;
            writer.WriteLine("buffer2D: " + name + " sliceIndex: " + sliceIndex);
            var arr = buffer.GetAs2DArray();
            var xBound = arr.GetUpperBound(0)+1; var yBound = arr.GetUpperBound(1)+1;
            writer.WriteLine("xBound: " + xBound + " yBound: "+ yBound);
            for (int x = 0; x < xBound; x++) {
                for(int y=0; y < yBound; y++) {
                    writer.Write(arr[x, y] + " ");
                }
            }            
        }
        /// <summary>
        /// Saves a 3DMemoryBuffer.
        /// </summary>
        protected void SaveBuffer(StreamWriter writer, MemoryBuffer3D<float> buffer, int sliceIndex, string name) {
            if (buffer.Equals(GPUHelper.dummyBuffer2D)) return;
            writer.WriteLine("buffer3D: " + name + " sliceIndex: " + sliceIndex);
            var arr = buffer.GetAs3DArray();
            var xBound = arr.GetUpperBound(0)+1; var yBound = arr.GetUpperBound(1)+1; var zBound = arr.GetUpperBound(1)+1;
            writer.WriteLine("xBound: " + xBound + " yBound: " + yBound + " zBound: " + zBound);
            for (int x = 0; x < xBound; x++) {
                for (int y = 0; y < yBound; y++) {
                    for (int z = 0; z < zBound; z++) {
                        writer.Write(arr[x, y, z] + " ");
                    }
                }
            }
        }
    }    
}
