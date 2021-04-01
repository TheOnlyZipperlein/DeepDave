using System;
using System.Collections.Generic;

namespace DeepDave.Helper {
    public static class MyExtensions {
        /// <summary>
        /// Shuffles the list randomly.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        public static void Shuffle<T>(this IList<T> list) {
            int n = list.Count;
            var rng = new Random(DateTime.Now.Millisecond);
            while (n > 1) {
                n--;                
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
