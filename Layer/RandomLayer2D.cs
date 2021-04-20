using DeepDave.Helper;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace DeepDave.Layer {
    public class RandomLayer2D : Layer2D {
        private int bestSlice;
        private MemoryBuffer3D<int>[] connectionInfo, newConnectionInfo;
        private RandomMatrixFactory conFactory, weightFactory;
        private MemoryBuffer3D<float>[] newWheight;
        private MemoryBuffer<float> averageError;
        internal new Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<int>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> sumCalculate;
        internal Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<float>> alternateWeights;
        internal Action<AcceleratorStream, Index2, ArrayView3D<int>, ArrayView3D<int>> alternateConnections;
        internal Action<AcceleratorStream, Index1, ArrayView2D<float>, ArrayView<float>> calculateAverageError;

        public RandomLayer2D(Layer2D prevLayer, int sliceCount, int maxConnectionsPerNeuron) : base(prevLayer, Layer2D.ActivationFunctions.ReLUActivation, sliceCount) {            
            var x = (int) prevLayer.GetActivatedBuffer(0).Extent.X;
            var y = (int) prevLayer.GetActivatedBuffer(0).Extent.Y;
            var fac = GetSuitableFactorForFunction(function, x * y);
            weightFactory = new RandomMatrixFactory(x, y, maxConnectionsPerNeuron, RandomMatrixFactory.GenerationType.Float);
            conFactory = new RandomMatrixFactory(x, y, maxConnectionsPerNeuron*2, RandomMatrixFactory.GenerationType.Integer, maxConnectionsPerNeuron);
            connectionInfo = new MemoryBuffer3D<int>[sliceCount];
            newConnectionInfo = new MemoryBuffer3D<int>[sliceCount];
            this.newWheight = new MemoryBuffer3D<float>[sliceCount];
            this.averageError = GPUHelper.CreateBuffer(sliceCount);
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, fac };
                this.variable[i] = GPUHelper.CreateBuffer(source, 2);
                
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, maxConnectionsPerNeuron);
                this.newWheight[i] = GPUHelper.CreateBuffer(x,y, maxConnectionsPerNeuron);
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
                bool b = true;
                while (b) if (conFactory.intQueue3D.Count > 1) b = false;
                int[,,] arr;
                conFactory.intQueue3D.TryDequeue(out arr);
                this.connectionInfo[i] = GPUHelper.CreateBuffer(arr, x, y, maxConnectionsPerNeuron*2);
                conFactory.intQueue3DUsed.Enqueue(arr);
                arr = null;
                conFactory.intQueue3D.TryDequeue(out arr);
                this.newConnectionInfo[i] = GPUHelper.CreateBuffer(arr , x, y, maxConnectionsPerNeuron*2);
                conFactory.intQueue3DUsed.Enqueue(arr);
            }
            var info = Type.GetType("DeepDave.Kernel.Randomfunctions").GetMethod(nameof(Kernel.Randomfunctions.AlternateConnections), BindingFlags.NonPublic | BindingFlags.Static);
            alternateConnections = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView3D<int>, ArrayView3D<int>>>();
            info = Type.GetType("DeepDave.Kernel.Randomfunctions").GetMethod(nameof(Kernel.Randomfunctions.AlternateWeights), BindingFlags.NonPublic | BindingFlags.Static);
            alternateWeights = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<float>>>();
            info = Type.GetType("DeepDave.Kernel.Randomfunctions").GetMethod(nameof(Kernel.Randomfunctions.Calculate), BindingFlags.NonPublic | BindingFlags.Static);
            sumCalculate = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView3D<int>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            info = Type.GetType("DeepDave.Kernel.Randomfunctions").GetMethod(nameof(Kernel.Randomfunctions.CalculateAverageError), BindingFlags.NonPublic | BindingFlags.Static);
            calculateAverageError = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index1, ArrayView2D<float>, ArrayView<float>>>();
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
            var b = true;
            bestSlice = int.MaxValue;
            for (int i = 0; i < activated.Length; i++) {
                while (b) if (conFactory.intQueue3D.Count > 1 && weightFactory.floatQueue3D.Count > 1) b = false;
                int[,,] arrI;
                while (!conFactory.intQueue3D.TryDequeue(out arrI));
                this.newConnectionInfo[i].CopyFrom(arrI, Index3.Zero, Index3.Zero, newConnectionInfo[i].Extent);
                conFactory.intQueue3DUsed.Enqueue(arrI);
                float[,,] arrF;
                while(!weightFactory.floatQueue3D.TryDequeue(out arrF));
                this.newWheight[i].CopyFrom(arrF, Index3.Zero, Index3.Zero, newWheight[i].Extent);
                weightFactory.floatQueue3DUsed.Enqueue(arrF);
            }
            ChangeConnections();
            ChangeWheights();
            GPUHelper.Call.Wait();
            for (int currentSlice = 1; currentSlice < activated.Length; currentSlice++) {
                sumCalculate(GPUHelper.accelerator.DefaultStream, this.GetUnactivatedBuffer(0).Extent, weight[currentSlice], connectionInfo[currentSlice], prevLayer.GetActivatedBuffer(currentSlice), sumInput[currentSlice], bias[currentSlice], variable[currentSlice]);
            }
        }
        internal override void AdjustWeights_() {
            
        }

        internal override void CalculateError_(MemoryBuffer2D<float>[] shoulds) {            
            for(int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                calculateAverageError(GPUHelper.accelerator.DefaultStream, (Index1) averageError.Extent, nextLayer.GetErrors(currentSlice), averageError);
            }
            GPUHelper.Call.Wait();
            var averages = averageError.GetAsArray();
            this.error = nextLayer.GetErros();
        }         
    }    
}
