using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System.Reflection;

namespace DeepDave.Helper {
    internal partial class GPUHelper {
        internal static MemoryBuffer2D<float> GetInputBuffer() {
            if (reusableInputBuffer.Count > 0) return reusableInputBuffer.Dequeue();
            return accelerator.Allocate<float>(Config.inputSize.Width, Config.inputSize.Height);
        }

        internal static void ScrapInputBuffer(MemoryBuffer2D<float> inputBuffer) {
            if (inputBuffer != null) reusableInputBuffer.Enqueue(inputBuffer);
        }
        internal static MemoryBuffer2D<float> GetOutputBuffer() {
            if (reusableOutputBuffer.Count > 0) return reusableOutputBuffer.Dequeue();
            return accelerator.Allocate<float>(Config.outputSize.Width, Config.outputSize.Height);
        }

        internal static void ScrapOutputBuffer(MemoryBuffer2D<float> outputBuffer) {
            if (outputBuffer != null) reusableOutputBuffer.Enqueue(outputBuffer);
        }

        internal static MemoryBuffer2D<float> CreateBuffer(int x, int y) {
            return CreateBuffer(RandomMatrixFactory.CreateRandomMatrix(x, y), x, y);
        }
        internal static MemoryBuffer3D<float> CreateBuffer(int x, int y, int z) {
            return CreateBuffer(RandomMatrixFactory.CreateRandomMatrix(x, y, z), x, y, z);
        }
        internal static MemoryBuffer<float> CreateBuffer(int x) {
            return CreateBuffer(RandomMatrixFactory.CreateRandomMatrix(x), x);
        }
        internal static MemoryBuffer<float> CreateBuffer(float[] source, int x) {
            var buffer = accelerator.Allocate<float>(x);
            buffers1DFloat.Add(buffer);
            buffer.CopyFrom(source, Index1.Zero, Index1.Zero, buffer.Extent);
            return buffer;
        }
        internal static MemoryBuffer2D<float> CreateBuffer(float[,] source, int x, int y) {
            var buffer = accelerator.Allocate<float>(x, y);
            buffers2DFloat.Add(buffer);
            buffer.CopyFrom(source, Index2.Zero, Index2.Zero, buffer.Extent);
            return buffer;
        }
        internal static MemoryBuffer3D<float> CreateBuffer(float[,,] source, int x, int y, int z) {
            var buffer = accelerator.Allocate<float>(x, y, z);
            buffers3DFloat.Add(buffer);
            buffer.CopyFrom(source, Index3.Zero, Index3.Zero, buffer.Extent);
            return buffer;
        }
        internal static MemoryBuffer<int> CreateBuffer(int[] source, int x) {
            var buffer = accelerator.Allocate<int>(x);
            buffers1DInt.Add(buffer);
            buffer.CopyFrom(source, Index1.Zero, Index1.Zero, buffer.Extent);
            return buffer;
        }
        internal static MemoryBuffer2D<int> CreateBuffer(int[,] source, int x, int y) {
            var buffer = accelerator.Allocate<int>(x, y);
            buffers2DInt.Add(buffer);
            buffer.CopyFrom(source, Index2.Zero, Index2.Zero, buffer.Extent);
            return buffer;
        }
        internal static MemoryBuffer3D<int> CreateBuffer(int[,,] source, int x, int y, int z) {
            var buffer = accelerator.Allocate<int>(x, y, z);
            buffers3DInt.Add(buffer);
            buffer.CopyFrom(source, Index3.Zero, Index3.Zero, buffer.Extent);
            return buffer;
        }

        internal static ILGPU.Runtime.Kernel CreateKernel(MethodInfo kernelFunction) {
            var entryPointDescCalculate = EntryPointDescription.FromImplicitlyGroupedKernel(kernelFunction);
            var compiledKernel = GPUHelper.accelerator.Backend.Compile(entryPointDescCalculate, KernelSpecialization.Empty);
            var kernel = GPUHelper.accelerator.LoadAutoGroupedKernel(compiledKernel);
            kernels.Add(kernel);
            return kernel;
        }
    }
}
