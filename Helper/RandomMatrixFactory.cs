using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Helper {
    public class RandomMatrixFactory {
        internal static float fixedfloat = 0.0f;
        internal static float dec = 1f;
        internal static Random rand = new Random(DateTime.Now.Millisecond);

        internal static float[] CreateRandomMatrix(int rows)
        {
            var matrix = new float[rows];

            for (var i = 0; i < rows; i++) {
                matrix[i] = GetNumber();
            }
            return matrix;
        }
        internal static float[,] CreateRandomMatrix(int rows, int columns) {     
            var matrix = new float[rows, columns];

            for (var i = 0; i < rows; i++) {
                for (var j = 0; j < columns; j++) {
                    matrix[i, j] = GetNumber();
                }
            }
            return matrix;
        }
        internal static float[,,] CreateRandomMatrix(int rows, int columns, int depth) {
            var matrix = new float[rows, columns, depth];

            for (var i = 0; i < rows; i++) {
                for (var j = 0; j < columns; j++) {
                    for (var k = 0; k < depth; k++) {
                        matrix[i,j,k] = GetNumber();
                    }
                }
            }
            return matrix;
        }
        internal static float GetNumber() {
            var number = fixedfloat;
            if (number != 0f) return number;
            number = dec * (float) (rand.NextDouble());
            if (rand.Next(2) % 2 == 0)
                number *= -1;
            return number;
        }
    }
}
