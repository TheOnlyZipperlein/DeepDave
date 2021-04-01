using DeepDave.Helper;
using DeepDave.Layer;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text;

namespace DeepDave {
    public class NetworkInput {
        internal float[][,] inputs, shoulds;
        private MemoryBuffer2D<float>[] inputsOnGPU, shouldsOnGPU;

        public NetworkInput(float[][,] inputs, float[][,] shoulds) {
            this.inputs = inputs;
            this.shoulds = shoulds;
            this.inputsOnGPU = new MemoryBuffer2D<float>[inputs.Length];
            this.shouldsOnGPU = new MemoryBuffer2D<float>[shoulds.Length];
        }

        public NetworkInput(byte[][,] inputs, byte[][,] shoulds) {
            this.inputs = new float[inputs.Length][,];
            this.shoulds = new float[shoulds.Length][,];
            this.inputsOnGPU = new MemoryBuffer2D<float>[inputs.Length];
            this.shouldsOnGPU = new MemoryBuffer2D<float>[shoulds.Length];

            for(int i=0; i<inputs.Length;i++) {
                this.inputs[i] = new float[inputs[i].GetUpperBound(0) + 1, inputs[i].GetUpperBound(1) + 1];
                System.Array.Copy(inputs[i], this.inputs[i], inputs[i].Length);
                this.shoulds[i] = new float[shoulds[i].GetUpperBound(0) + 1, shoulds[i].GetUpperBound(1) + 1];
                System.Array.Copy(shoulds[i], this.shoulds[i], shoulds[i].Length);
                for(int x=0; x < 28; x++) {
                    for(int y=0; y<28; y++) {
                        this.inputs[0][x, y] /= 255;
                    }
                }
            }


        }

        public void Load() {
            for (int i = 0; i < inputsOnGPU.Length; i++) {
                inputsOnGPU[i] = GPUHelper.GetInputBuffer();
                inputsOnGPU[i].CopyFrom(inputs[i], Index2.Zero, Index2.Zero, inputsOnGPU[i].Extent);
            }            
        }

        public void Unload() {
            for (int i = 0; i < inputsOnGPU.Length; i++) {
                GPUHelper.ScrapInputBuffer(inputsOnGPU[i]);
            }
        }
        public MemoryBuffer2D<float>[] GetInputs() {
            return this.inputsOnGPU;
        }

        public void LoadShoulds() {
            for(int i=0; i<shoulds.Length; i++) {
                shouldsOnGPU[i] = GPUHelper.GetOutputBuffer();
                shouldsOnGPU[i].CopyFrom(this.shoulds[i], Index2.Zero, Index2.Zero, shouldsOnGPU[i].Extent);
            }
        }


        internal MemoryBuffer2D<float>[] GetShouldsActivated(InputLayer2D inputLayer) {
            LoadShoulds();
            var buffer = inputLayer.SwapInputs(shouldsOnGPU);
            inputLayer.CalculateOutput();
            for (int i = 0; i < buffer.Length; i++) {
                GPUHelper.ScrapOutputBuffer(buffer[i]);  
            }
            return inputLayer.GetActivatedBuffer();
        }
    }
}
