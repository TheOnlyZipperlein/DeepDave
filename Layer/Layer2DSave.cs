using DeepDave.Helper;
using DeepDave.Helper.AbstractionClasses;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeepDave.Layer {
    public abstract partial class Layer2D : Saveable {

        void Saveable.Save(StreamWriter writer) {
            writer.WriteLine("Type: " + this.GetType().FullName);
            writer.WriteLine("Function: " + this.function);
            int sliceCount = this.bias.Length;
            writer.WriteLine("SliceCount: " + this.bias.Length);            
            for (int i = 0; i < sliceCount; i++) {
                writer.WriteLine("buffer2D: bias sliceIndex: " + i);
                if (!bias[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, bias[i]);
                }
                writer.WriteLine("buffer2D: derived sliceIndex: " + i);
                if (!derived[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, derived[i]);
                }
                writer.WriteLine("buffer2D: errors sliceIndex: " + i);
                if (!errors[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, errors[i]);
                }
                writer.WriteLine("buffer2D: activated sliceIndex: " + i);
                if (!activated[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, activated[i]);
                }
                writer.WriteLine("buffer2D: sumInput sliceIndex: " + i);
                if (!sumInput[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, sumInput[i]);
                }
                writer.WriteLine("buffer2D: weights sliceIndex: " + i);
                if (!weights[i].Equals(GPUHelper.dummyBuffer2D)) {
                    this.SaveBuffer(writer, weights[i]);
                }
            }
        }

        protected void SaveBuffer(StreamWriter writer, MemoryBuffer2D<float> buffer) {
            var arr = buffer.GetAs2DArray();
            var xBound = arr.GetUpperBound(0)+1; var yBound = arr.GetUpperBound(1)+1;
            writer.WriteLine("xBound: " + xBound + " yBound: "+ yBound);
            for (int x = 0; x < xBound; x++) {
                for(int y=0; y < yBound; y++) {
                    writer.Write(arr[x, y] + " ");
                }
            }            
        }

        protected void SaveBuffer(StreamWriter writer, MemoryBuffer3D<float> buffer) {
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
