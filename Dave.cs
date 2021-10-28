using DeepDave.Helper;
using DeepDave.Helper.AbstractionClasses;
using DeepDave.Helper.Exceptions;
using DeepDave.Layer;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DeepDave {
    public class Dave {
        internal List<Layer2D> layers;
        internal InputLayer2D shouldActivationLayer;
        internal List<NetworkInput> usedInputs;
        internal Queue<NetworkInput> inputQueue;
        internal List<float> listOfRatios;

        /// <summary>
        /// Creates a new Layer.
        /// </summary>
        /// <param name="inputSize"></param>
        public Dave(Size inputSize, int sliceCount = 1, bool debuggingToggle = false) {
            PreInit();
            Config.inputSize = inputSize;
            Config.DebuggingToggle = debuggingToggle;
            GPUHelper.CreateAccelerator(debuggingToggle);
            layers.Add(new InputLayer2D(inputSize, null, sliceCount, "ByteToByteFraction"));
        }

        public Dave(string pathToFile) {
            PreInit();
            StreamReader reader = new StreamReader(pathToFile);
            while (!reader.EndOfStream) {
                var line = reader.ReadLine();

            }
        }

        public void PreInit() {
            listOfRatios = new List<float>();
            layers = new List<Layer2D>();
            inputQueue = new Queue<NetworkInput>();
            usedInputs = new List<NetworkInput>();
        }

        public void Init() {
            foreach (Layer2D layer in layers) {
                layer.Init();
            }
            Config.outputSize = new Size((int)layers.Last().GetActivatedBuffer(0).Width, (int)layers.Last().GetActivatedBuffer(0).Height);
            shouldActivationLayer = new InputLayer2D(Config.outputSize, null, layers.Last().GetActivatedBuffer().Length, "ByteToByteFraction");
        }

        /// <summary>
        /// Add an input to the queue for async calculating.
        /// </summary>
        /// <param name="input"></param>
        public void AddInput(NetworkInput input) {
            inputQueue.Enqueue(input);
        }

        public void AddInput(List<NetworkInput> inputs) {
            foreach (NetworkInput input in inputs) {
                inputQueue.Enqueue(input);
            }
        }

        /// <summary>
        /// Enables learning, to disable learning th e network has to be saved and loaded.
        /// </summary>
        public void EnableLearning() {
            Config.learningEnabled = true;
        }

        public void Save(String path) {
            if (File.Exists(path)) File.Delete(path);
            var writer = new StreamWriter(File.OpenWrite(path));
            foreach (Layer2D layer in layers) {
                ((Saveable)layer).Save(writer);
            }
            writer.Flush();
            writer.Close();
        }

        public Layer2D GetLastLayer() {
            return layers.Last();
        }

        /// <summary>
        /// Adds a layer to the network.
        /// </summary>
        /// <param name="layer"></param>
        public void AddLayer(Layer2D layer) {
            layers.Add(layer);
        }

        /// <summary>
        /// Shuffles and reuses configured Inputs.
        /// </summary>
        private void ReuseNetworkInputs() {
            usedInputs.Shuffle();
            foreach (NetworkInput momInput in usedInputs) {
                inputQueue.Enqueue(momInput);
            }
            usedInputs.Clear();
        }

        /// <summary>
        /// Call to calculate the output for the first input of the inputQueue.
        /// </summary>
        /// <returns>The output of the current input values.</returns>
        private float[][,] CalculateInput() {
            if (Config.learningEnabled & inputQueue.Count == 0 & usedInputs.Count > 0 & Config.KeepDataForNewEpoches) ReuseNetworkInputs();
            var input = inputQueue.Dequeue();
            input.Load();
            ((InputLayer2D)layers.ElementAt(0)).SwapInputs(input.GetInputs());
            MemoryBuffer2D<float>[] activatedShoulds = null;
            if (Config.learningEnabled) activatedShoulds = input.GetShouldsActivated(shouldActivationLayer);
            layers.Last().CalculateOutput(activatedShoulds);
            usedInputs.Add(input);
            input.Unload();

            var re = new float[layers.Last().GetActivatedBuffer().Length][,];
            for (int i = 0; i < re.Length; i++) {
                re[i] = layers.Last().GetActivatedBuffer(i).GetAs2DArray();
            }
            done++;
            var right = 0;

            var pred = 0;
            float max = re[0][0, 0];
            for (int i = 0; i < 10; i++) {
                if (input.shoulds[0][i, 0] == 1f) right = i;
                if (max < re[0][i, 0]) {
                    max = re[0][i, 0];
                    pred = i;
                }
            }
            if (pred == right) counter++;
            var ratio = (float)counter / (float)done;
            var maxS = re[0][right, 0];
            return re;
        }

        /// <summary>
        /// Call to process a single value.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public float[][,] CalculateInput(NetworkInput input) {
            if (inputQueue.Count == 0) {// & Config.learningEnabled == false) {
                inputQueue.Enqueue(input);
                var output = CalculateInput();
                return output;
            } else {
                throw new IllegalCallException("CalculateInput is not valid during learning or if a input is already in Queue. " +
                "Please restart the network to use this operation.");
            }

        }

        public void Reset() {
            while (inputQueue.Count > 0) {
                inputQueue.Dequeue();
            }
        }

        int done = 0, counter = 0;
        public void DoEpoche() {
            done = 0; counter = 0; int percent = 0;
            var epocheTime = DateTime.Now;
            while (inputQueue.Count > 0) {
                DateTime now = DateTime.Now;
                CalculateInput();
                var time = DateTime.Now.Subtract(now).TotalMilliseconds;
                int newPercent = done * 100 / (done + inputQueue.Count);
                if (newPercent != percent) {
                    percent = newPercent;
                    //Console.WriteLine(percent + "% done of Epoche "+ listOfRatios.Count+1 + ".");
                }
            }
            listOfRatios.Add((float)counter / (float)done);

            if (listOfRatios.Count >= 2) Console.WriteLine("Epoche: " + listOfRatios.Count + " Ratio: " + listOfRatios[listOfRatios.Count - 1] + " Change: " + (listOfRatios[listOfRatios.Count - 1] - listOfRatios[listOfRatios.Count - 2]) + "Seconds Needed: " + DateTime.Now.Subtract(epocheTime).TotalSeconds);

            ReuseNetworkInputs();
        }
    }
}
