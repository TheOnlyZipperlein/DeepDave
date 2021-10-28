using DeepDave.Logging;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepDave.Helper {
    internal partial class GPUHelper {
        internal static Context context;
        internal static Accelerator accelerator;
        internal static Queue<MemoryBuffer2D<float>> reusableInputBuffer, reusableOutputBuffer;
        internal static List<Kernel> kernels;
        internal static List<MemoryBuffer<float>> buffers1DFloat;
        internal static List<MemoryBuffer2D<float>> buffers2DFloat;
        internal static List<MemoryBuffer3D<float>> buffers3DFloat;
        internal static List<MemoryBuffer<int>> buffers1DInt;
        internal static List<MemoryBuffer2D<int>> buffers2DInt;
        internal static List<MemoryBuffer3D<int>> buffers3DInt;
        internal static MemoryBuffer<float> dummyBuffer1D;
        internal static MemoryBuffer2D<float> dummyBuffer2D;
        internal static MemoryBuffer3D<float> dummyBuffer3D;

        internal static void CreateAccelerator(Boolean debugMode) {
            context = new Context(
                ContextFlags.EnableParallelCodeGenerationInFrontend
                );
            context.EnableAlgorithms();
            List<Accelerator> accelerators = new List<Accelerator>();

            Accelerator cpuAccl = null;
            foreach (var acceleratorId in Accelerator.Accelerators) {
                accelerator = Accelerator.Create(context, acceleratorId);
                accelerators.Add(accelerator);
                if (accelerator.AcceleratorType == AcceleratorType.CPU) {
                    cpuAccl = accelerator;
                }
            }
            for (int i = 0; i < accelerators.Count; i++) {
                var accl = accelerators.ElementAt(i);
                if (cpuAccl != accl) accelerator = accl;
            }
            if (accelerator == null) throw new Exception("No compitable device found.");
            if (debugMode) accelerator = cpuAccl;
            if (debugMode) Logger.WriteLine("Debug Mode - Enabled");
            if (accelerator == cpuAccl) Logger.WriteLine("Performing operations on CPU Accelerator");
            else Logger.WriteLine("Performing operations on " + accelerator.Name);

            reusableInputBuffer = new Queue<MemoryBuffer2D<float>>();
            reusableOutputBuffer = new Queue<MemoryBuffer2D<float>>();
            kernels = new List<Kernel>();
            buffers1DInt = new List<MemoryBuffer<int>>();
            buffers2DInt = new List<MemoryBuffer2D<int>>();
            buffers3DInt = new List<MemoryBuffer3D<int>>();
            buffers1DFloat = new List<MemoryBuffer<float>>();
            buffers2DFloat = new List<MemoryBuffer2D<float>>();
            buffers3DFloat = new List<MemoryBuffer3D<float>>();
            dummyBuffer1D = CreateBuffer(1);
            dummyBuffer2D = GPUHelper.accelerator.Allocate<float>(1, 1);
            dummyBuffer3D = GPUHelper.accelerator.Allocate<float>(1, 1, 1);
        }
    }
}
