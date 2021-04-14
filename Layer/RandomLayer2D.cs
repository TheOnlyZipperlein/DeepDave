using DeepDave.Helper;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepDave.Layer {
    public class RandomLayer2D : Layer2D {
        private MemoryBuffer3D<float>[] connectionInfo;

        public RandomLayer2D(Layer2D prevLayer, int sliceCount, int maxConnectionsPerNeuron) : base(prevLayer, null, sliceCount) {
            var x = (int) prevLayer.GetActivatedBuffer(0).Extent.X;
            var y = (int) prevLayer.GetActivatedBuffer(0).Extent.Y;
            var fac = GetSuitableFactorForFunction(function, x * y);
            connectionInfo = new MemoryBuffer3D<float>[sliceCount];
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, fac };
                this.variable[i] = GPUHelper.CreateBuffer(source, 2);
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, maxConnectionsPerNeuron);
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
                this.connectionInfo[i] = GPUHelper.CreateBuffer(x, y, maxConnectionsPerNeuron);
            }            
        }

        protected override void CreateKernelInfos() {
            //
        }

        internal override void CalculateOutput_() {
            
        }

    }
    }
}
