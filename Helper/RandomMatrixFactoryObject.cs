using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace DeepDave.Helper {
    internal partial class RandomMatrixFactory {
        internal ConcurrentQueue<float[,,]> floatQueue3D;
        internal ConcurrentQueue<int[,,]> intQueue3D;
        internal ConcurrentQueue<float[,]> floatQueue2D;
        internal ConcurrentQueue<int[,]> intQueue2D;
        internal ConcurrentQueue<float[]> floatQueue1D;
        internal ConcurrentQueue<int[]> intQueue1D;

        internal ConcurrentQueue<float[,,]> floatQueue3DUsed;
        internal ConcurrentQueue<int[,,]> intQueue3DUsed;
        internal ConcurrentQueue<float[,]> floatQueue2DUsed;
        internal ConcurrentQueue<int[,]> intQueue2DUsed;
        internal ConcurrentQueue<float[]> floatQueue1DUsed;
        internal ConcurrentQueue<int[]> intQueue1DUsed;

        private Random objRand;
        private GenerationType type;
        private int x, y, z;
        private int maxInteger;

        internal RandomMatrixFactory(int x, int y, int z, GenerationType type, int maxInteger = 0) {
            this.type = type;
            this.x = x;
            this.y = y;
            this.z = z;
            this.objRand = new Random(DateTime.Now.Millisecond);
            Thread trd = new Thread(Loop);
            this.maxInteger = maxInteger;
            switch (type) {
                case GenerationType.Float:
                    if (z == 0) {
                        if (y == 0) {
                            floatQueue1D = new ConcurrentQueue<float[]>();
                            floatQueue1DUsed = new ConcurrentQueue<float[]>();
                        } else {
                            floatQueue2D = new ConcurrentQueue<float[,]>();
                            floatQueue2DUsed = new ConcurrentQueue<float[,]>();
                        }
                    } else {
                        floatQueue3D = new ConcurrentQueue<float[,,]>();
                        floatQueue3DUsed = new ConcurrentQueue<float[,,]>();
                    } break;
                case GenerationType.Integer:
                    if (z == 0) {
                        if (y == 0) {
                            intQueue1D = new ConcurrentQueue<int[]>();
                            intQueue1DUsed = new ConcurrentQueue<int[]>();
                        } else {
                            intQueue2D = new ConcurrentQueue<int[,]>();
                            intQueue2DUsed = new ConcurrentQueue<int[,]>();
                        }
                    } else {
                        intQueue3D = new ConcurrentQueue<int[,,]>();
                        intQueue3DUsed = new ConcurrentQueue<int[,,]>();
                    } break;
            }
            trd.Start();
        }

        internal void Loop() {
            while (true) {
                if (QueueNotFull()) {
                    if (z == 0) {
                        if (y == 0) {
                            Add1D();
                        } else Add2D();
                    } else Add3D();
                } else Thread.Sleep(1);
            }
        }
        internal void Add1D() {
            switch (type) {
                case GenerationType.Float:
                    float[] arr = null; 
                    floatQueue1DUsed.TryDequeue(out arr);
                    if(arr==null) arr = new float[this.x];
                    for (int x = 0; x < this.x; x++) {
                        if (objRand.NextDouble() > 0.99)
                            arr[x] = (float)objRand.NextDouble();
                    }
                    floatQueue1D.Enqueue(arr);
                    break;
                case GenerationType.Integer:
                    int[] arrI = null;
                    intQueue1DUsed.TryDequeue(out arrI);
                    if (arrI == null) arrI = new int[this.x];
                    for (int x = 0; x < this.x; x++) {
                        if (objRand.NextDouble() > 0.99)
                            arrI[x] = (int)objRand.Next(0, maxInteger);
                    }
                    intQueue1D.Enqueue(arrI);
                    break;
            }
        }
        internal void Add2D() {
            switch (type) {
                case GenerationType.Float:
                    float[,] arrF = null;
                    floatQueue2DUsed.TryDequeue(out arrF);
                    if (arrF == null) arrF = new float[this.x, this.y];
                    for (int x = 0; x < this.x; x++) {
                        for (int y = 0; y < this.y; y++) {
                            if (objRand.NextDouble() > 0.99)
                                arrF[x, y] = (float)objRand.NextDouble();
                        }
                    }
                    floatQueue2D.Enqueue(arrF);
                    break;
                case GenerationType.Integer:
                    int[,] arrI = null;
                    intQueue2DUsed.TryDequeue(out arrI);
                    if (arrI == null) arrI = new int[this.x, this.y];
                    for (int x = 0; x < this.x; x++) {
                        for (int y = 0; y < this.y; y++) {
                            if (objRand.NextDouble() > 0.99)
                                arrI[x, y] = (int)objRand.Next(0, maxInteger);
                        }
                    }
                    intQueue2D.Enqueue(arrI);
                    break;
            }
        }
        internal void Add3D() {
            switch (type) {
                case GenerationType.Float:
                    float[,,] arr = null;
                    floatQueue3DUsed.TryDequeue(out arr);
                    if (arr == null) arr = new float[this.x, this.y, this.z];
                    for (int x = 0; x < this.x; x++) {
                        for (int y = 0; y < this.y; y++) {
                            for (int z = 0; z < this.z; z++) {
                                if (objRand.NextDouble() > 0.99)
                                    arr[x, y, z] = (float)objRand.NextDouble();
                            }
                        }
                    }
                    floatQueue3D.Enqueue(arr);
                    break;
                case GenerationType.Integer:
                    int[,,] arrI = null;
                    intQueue3DUsed.TryDequeue(out arrI);
                    if (arrI == null) arrI = new int[this.x, this.y, this.z];
                    for (int x = 0; x < this.x; x++) {
                        for (int y = 0; y < this.y; y++) {
                            for (int z = 0; z < this.z; z++) {
                                if (objRand.NextDouble() > 0.99)
                                    arrI[x, y, z] = (int)objRand.Next(0, maxInteger);
                            }
                        }
                    }
                    intQueue3D.Enqueue(arrI);
                    break;
            }
        }
        internal bool QueueNotFull() {
            switch (type) {
                case GenerationType.Float:
                    if (floatQueue1D != null && floatQueue1D.Count < 10) return true;
                    if (floatQueue2D != null && floatQueue2D.Count < 10) return true;
                    if (floatQueue3D != null && floatQueue3D.Count < 10) return true;
                    break;
                case GenerationType.Integer:
                    if (intQueue1D != null && intQueue1D.Count < 10) return true;
                    if (intQueue2D != null && intQueue2D.Count < 10) return true;
                    if (intQueue3D != null && intQueue3D.Count < 10) return true;
                    break;
            }
            return false;
        }
    }
}
