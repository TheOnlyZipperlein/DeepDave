using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Layer {
    public abstract partial class Layer2D {
        internal virtual MemoryBuffer2D<float> GetUnactivatedBuffer(int currentSlice) {
            return sumInput[currentSlice];
        }
        internal virtual MemoryBuffer2D<float>[] GetUnactivatedBuffer() {
            return sumInput;
        }
        internal virtual MemoryBuffer2D<float> GetActivatedBuffer(int currentSlice) {
            return activated[currentSlice];

        }
        internal virtual MemoryBuffer2D<float>[] GetActivatedBuffer() {
            return activated;
        }
        internal virtual MemoryBuffer3D<float> GetWeightBuffer(int currentSlice) {
            return weight[currentSlice];
        }
        internal virtual MemoryBuffer3D<float>[] GetWeightBuffer() {
            return weight;
        }
        internal virtual MemoryBuffer2D<float> GetErrors(int currentSlice) {
            return error[currentSlice];
        }
        internal virtual MemoryBuffer2D<float> GetDerived(int currentSlice) {
            return derived[currentSlice];
        }
        internal virtual MemoryBuffer2D<float>[] GetErros() {
            return error;
        }
        internal virtual MemoryBuffer2D<float>[] GetDerived() {
            return derived;
        }

        internal virtual Layer2D GetNextLayter() {
            return nextLayer;
        }
        internal virtual Layer2D GetPrevious() {
            return prevLayer;
        }
    }
}

