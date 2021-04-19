using DeepDave.Helper;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepDave.Layer {
    public class RandomLayer2D : Layer2D {
        private MemoryBuffer3D<int>[] connectionInfo, newConnectionInfo;
        private RandomMatrixFactory factory;
        private MemoryBuffer3D<float>[] newWheight;
        internal new Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<int>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> sumCalculate;
        internal Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<float>> alternateWeights;
        internal Action<AcceleratorStream, Index2, ArrayView3D<int>, ArrayView3D<int>> alternateConnections;

        public RandomLayer2D(Layer2D prevLayer, int sliceCount, int maxConnectionsPerNeuron) : base(prevLayer, null, sliceCount) {
            var x = (int) prevLayer.GetActivatedBuffer(0).Extent.X;
            var y = (int) prevLayer.GetActivatedBuffer(0).Extent.Y;
            var fac = GetSuitableFactorForFunction(function, x * y);
            var weightFactory = new RandomMatrixFactory(x, y, maxConnectionsPerNeuron, RandomMatrixFactory.GenerationType.Float);
            var conFactory = new RandomMatrixFactory(x, y, maxConnectionsPerNeuron*2, RandomMatrixFactory.GenerationType.Integer, maxConnectionsPerNeuron);
            connectionInfo = new MemoryBuffer3D<int>[sliceCount];
            newConnectionInfo = new MemoryBuffer3D<int>[sliceCount];            
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, fac };
                this.variable[i] = GPUHelper.CreateBuffer(source, 2);
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, maxConnectionsPerNeuron);
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
                bool b = true;
                while (b) if (conFactory.intQueue3D.Count > 1) b = false;
                int[,,] arr;
                conFactory.intQueue3D.TryDequeue(out arr);
                this.connectionInfo[i] = GPUHelper.CreateBuffer(arr, x, y, maxConnectionsPerNeuron*2);
                conFactory.intQueue3D.TryDequeue(out arr);
                this.newConnectionInfo[i] = GPUHelper.CreateBuffer(arr , x, y, maxConnectionsPerNeuron*2);
            }
            var extent = connectionInfo[0].Extent;
        }

        internal void ChangeConnections() {
            for (int currentSlice = 1; currentSlice < activated.Length; currentSlice++) {
                alternateConnections(GPUHelper.accelerator.DefaultStream, activated[0].Extent, connectionInfo[currentSlice], newConnectionInfo[currentSlice]);
            }
        }

        internal void ChangeWheights() {
            for (int currentSlice = 1; currentSlice < activated.Length; currentSlice++) {
                alternateWeights(GPUHelper.accelerator.DefaultStream, activated[0].Extent, weight[currentSlice], newWheight[currentSlice]);
            }
        }
        internal override void CalculateOutput_() {
            ChangeConnections();
            ChangeWheights();
            GPUHelper.Call.Wait();
        }
        internal override void AdjustWeights_() {
            
        }

        internal override void CalculateError_(MemoryBuffer2D<float>[] shoulds) {
            
        } 
    }    
}
