using DeepDave.Helper;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Reflection;

namespace DeepDave.Layer {
    public abstract partial class Layer2D {
        protected Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> sumCalculate;
        protected Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> sumForError;
        protected Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> adjustWheigts;
        protected Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> activation;
        protected Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> derivation;

        protected MethodInfo activationFunction, derivationFunction, sumCalculateFunction, sumErrorFunction, adjustWheigtsFunction;
        internal string function;

        protected MemoryBuffer<float>[] variable;
        protected MemoryBuffer2D<float>[] bias, derived, error, activated, sumInput;
        protected MemoryBuffer3D<float>[] weight;

        protected Layer2D prevLayer, nextLayer;
        protected Layer2D(Layer2D prevLayer, string activationFunction, int sliceCount) {
            SetPrevious(prevLayer);
            if (prevLayer != null) prevLayer.SetNext(this);
            this.function = activationFunction;

            variable = new MemoryBuffer<float>[sliceCount];
            activated = new MemoryBuffer2D<float>[sliceCount];
            sumInput = new MemoryBuffer2D<float>[sliceCount];
            error = new MemoryBuffer2D<float>[sliceCount];
            weight = new MemoryBuffer3D<float>[sliceCount];
            bias = new MemoryBuffer2D<float>[sliceCount];
            derived = new MemoryBuffer2D<float>[sliceCount];
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, 0f };
                this.variable[i] = GPUHelper.CreateBuffer(source, source.Length);
                activated[i] = GPUHelper.dummyBuffer2D;
                sumInput[i] = GPUHelper.dummyBuffer2D;
                error[i] = GPUHelper.dummyBuffer2D;
                weight[i] = GPUHelper.dummyBuffer3D;
                bias[i] = GPUHelper.dummyBuffer2D;
                derived[i] = GPUHelper.dummyBuffer2D;
            }
        }
        internal void Init() {
            this.CreateKernelInfos();
            this.InitializeKernels();
        }
        protected virtual void CreateKernelInfos() {
            this.CreateCalculateKernelInfo();
            this.CreateAdjustWeightsKernelInfo();
            this.CreateCalculateErrorKernelInfo();
            this.CreateActivationFunctionInfo();
        }
        protected virtual void InitializeKernels() {
            if (sumCalculateFunction != null) {
                sumCalculate = GPUHelper.CreateKernel(sumCalculateFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
            if (sumErrorFunction != null) {
                if (nextLayer != null) sumForError = GPUHelper.CreateKernel(sumErrorFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
                else sumForError = GPUHelper.CreateKernel(sumErrorFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
            if (adjustWheigtsFunction != null) {
                adjustWheigts = GPUHelper.CreateKernel(adjustWheigtsFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
            if (activationFunction != null) {
                activation = GPUHelper.CreateKernel(activationFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
            if (derivationFunction != null) {
                derivation = GPUHelper.CreateKernel(derivationFunction).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
        }

        internal virtual void CreateActivationFunctionInfo() {
            if (function != null) {
                if (activationFunction == null) activationFunction = Type.GetType("DeepDave.Layer.Kernels.ActivationFunctions").GetMethod(function, BindingFlags.NonPublic | BindingFlags.Static);
                if (derivationFunction == null) derivationFunction = Type.GetType("DeepDave.Layer.Kernels.DerivativeFunctions").GetMethod(function, BindingFlags.NonPublic | BindingFlags.Static);
            }
        }

        internal virtual void CreateCalculateKernelInfo() {
            if (sumCalculateFunction == null) sumCalculateFunction = Type.GetType("DeepDave.Layer.Kernels.SumCalculate").GetMethod(this.GetType().Name, BindingFlags.NonPublic | BindingFlags.Static);
        }
        internal virtual void CreateCalculateErrorKernelInfo() {
            if (Config.learningEnabled & sumErrorFunction == null) {
                if (nextLayer == null) sumErrorFunction = Type.GetType("DeepDave.Layer.Kernels.SumErrorOut").GetMethod(nameof(Kernels.SumErrorOut.Common), BindingFlags.NonPublic | BindingFlags.Static);
                else sumErrorFunction = Type.GetType("DeepDave.Layer.Kernels.SumErrorHidden").GetMethod(nextLayer.GetType().Name, BindingFlags.NonPublic | BindingFlags.Static);
            }
        }
        internal virtual void CreateAdjustWeightsKernelInfo() {
            if (Config.learningEnabled & sumErrorFunction == null) {
                var type = Type.GetType("DeepDave.Layer.Kernels.WheightAdjustment");
                var name = this.GetType().Name;
                adjustWheigtsFunction = type.GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static);
            }
        }
    }
}
