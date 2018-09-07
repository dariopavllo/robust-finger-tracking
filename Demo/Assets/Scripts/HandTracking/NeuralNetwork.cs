using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AMath = Accord.Math;

namespace HandTracking
{
    /// <summary>
    /// Implements a lightweight implementation of a neural network, whose weights can be imported from a file.
    /// Supports Dense, SimpleRNN and LSTM layers.
    /// </summary>
    public class NeuralNetwork
    {
        private readonly List<Layer> layers;

        /// <summary>
        /// Initialize a neural network.
        /// </summary>
        /// <param name="fileName">The file name of the model to import.</param>
        public NeuralNetwork(string fileName)
        {
            layers = new List<Layer>();
            var file = new BinaryReader(new FileStream(fileName, FileMode.Open));

            int numLayers = file.ReadInt32();
            for (int i = 0; i < numLayers; i++)
            {
                int layerType = file.ReadInt32();
                Layer layer;
                switch (layerType)
                {
                    case 0:
                    case 3:
                        layer = new Dense(file);
                        break;
                    case 1:
                        layer = new SimpleRNN(file);
                        break;
                    case 2:
                        layer = new LSTM(file);
                        break;
                    default:
                        throw new ArgumentException("Unsupported/invalid layer type");
                }

                layers.Add(layer);
            }

            file.Close();
        }

        /// <summary>
        /// Predicts the output given the input.
        /// </summary>
        /// <param name="input">The data to be fed to the input layer.</param>
        /// <returns>The result of the output layer.</returns>
        public float[] Predict(float[] input)
        {
            UnityEngine.Profiling.Profiler.BeginSample("NNPredict");
            foreach (Layer layer in layers)
            {
                input = layer.FeedForward(input);
            }
            UnityEngine.Profiling.Profiler.EndSample();
            return input;
        }

        /// <summary>
        /// Tells whether this neural network is recurrent (i.e. contains recurrent layers).
        /// </summary>
        /// <returns>True if this neural network contains recurrent layers</returns>
        public bool IsRecurrent()
        {
            foreach (Layer layer in layers)
            {
                if (layer.IsRecurrent())
                {
                    return true;
                }
            }
            return false;
        }

        private abstract class Layer
        {
            public delegate float ActivationFunction(float x);

            public abstract float[] FeedForward(float[] input);
            public abstract bool IsRecurrent();

            protected static float[,] ReadMatrix(BinaryReader file)
            {
                int shape1 = file.ReadInt32();
                int shape2 = file.ReadInt32();
                float[,] matrix = new float[shape1, shape2];
                for (int j = 0; j < shape1; j++)
                {
                    for (int k = 0; k < shape2; k++)
                    {
                        matrix[j, k] = file.ReadSingle();
                    }
                }
                return matrix;
            }

            protected static float[] ReadVector(BinaryReader file)
            {
                int shape = file.ReadInt32();
                float[] vector = new float[shape];
                for (int k = 0; k < shape; k++)
                {
                    vector[k] = file.ReadSingle();
                }
                return vector;
            }

            protected static ActivationFunction GetActivationFunction(int id)
            {
                switch (id)
                {
                    case 0:
                        return ActivationFunctions.ReLU;
                    case 1:
                        return ActivationFunctions.Linear;
                    case 2:
                        return ActivationFunctions.Sigmoid;
                    case 3:
                        return ActivationFunctions.Tanh;
                    case 4:
                        return ActivationFunctions.HardSigmoid;
                    default:
                        throw new ArgumentException("Unsupported/invalid activation function");
                }
            }

            protected static float[] ComputeActivations(float[] x, ActivationFunction func, bool inPlace = false)
            {
                if (!inPlace)
                {
                    x = (float[])x.Clone();
                }
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = func(x[i]);
                }
                return x;
            }
        }

        /// <summary>
        /// Represents a fully connected layer (class Dense in Keras).
        /// </summary>
        private class Dense : Layer
        {
            private readonly float[,] weights;
            private readonly float[] biases;
            private readonly ActivationFunction func;
            private readonly float[] outputBuffer;

            public Dense(BinaryReader file)
            {
                func = GetActivationFunction(file.ReadInt32());
                // Transpose the weight matrix (more cache-efficient matrix multiplication)
                weights = AMath.Matrix.Transpose(ReadMatrix(file));
                biases = ReadVector(file);

                // Pre-allocate memory for matrix multiplication in order to avoid unnecessary garbage collection
                outputBuffer = new float[biases.Length];
            }

            public override float[] FeedForward(float[] input)
            {
                AMath.Matrix.Dot(weights, input, outputBuffer);
                AMath.Elementwise.Add(outputBuffer, biases, outputBuffer);
                return ComputeActivations(outputBuffer, func, true);
            }

            public override bool IsRecurrent()
            {
                return false;
            }
        }

        /// <summary>
        /// Represents an Elman RNN layer (class SimpleRNN in Keras).
        /// </summary>
        private class SimpleRNN : Layer
        {
            private readonly float[,] weights;
            private readonly float[,] recurrentWeights;
            private readonly float[] biases;
            private readonly ActivationFunction func;
            private float[] state;

            public SimpleRNN(BinaryReader file)
            {
                func = GetActivationFunction(file.ReadInt32());
                weights = ReadMatrix(file);
                recurrentWeights = ReadMatrix(file);
                biases = ReadVector(file);

                state = new float[biases.Length];
            }

            public override float[] FeedForward(float[] input)
            {
                float[] result = AMath.Matrix.Dot(input, weights);
                float[] recurrentResult = AMath.Matrix.Dot(state, recurrentWeights);
                result = AMath.Elementwise.Add(AMath.Elementwise.Add(result, recurrentResult), biases);
                state = ComputeActivations(result, func, true);
                return state;
            }

            public override bool IsRecurrent()
            {
                return true;
            }
        }

        /// <summary>
        /// Represents a long short-term memory layer.
        /// </summary>
        private class LSTM : Layer
        {
            // Input, Forget, Cell, Output
            private readonly float[,] Wi, Wf, Wc, Wo; // Weights
            private readonly float[,] Ui, Uf, Uc, Uo; // Recurrent weights
            private readonly float[] bi, bf, bc, bo; // Biases
            private readonly ActivationFunction func;

            // Internal state
            private float[] h, c;

            public LSTM(BinaryReader file)
            {
                func = GetActivationFunction(file.ReadInt32());
                float[,] kernel = ReadMatrix(file);
                float[,] recurrentKernel = ReadMatrix(file);
                float[] biases = ReadVector(file);

                int units = biases.Length / 4;
                h = new float[units];
                c = new float[units];
                for (int i = 0; i < 4; i++)
                {
                    int start = i * units;

                    // Kernel
                    float[,] res = new float[kernel.GetLength(0), units];
                    for (int j = 0; j < kernel.GetLength(0); j++)
                    {
                        for (int k = 0; k < units; k++)
                        {
                            res[j, k] = kernel[j, start + k];
                        }
                    }
                    switch (i)
                    {
                        case 0:
                            Wi = res;
                            break;
                        case 1:
                            Wf = res;
                            break;
                        case 2:
                            Wc = res;
                            break;
                        case 3:
                            Wo = res;
                            break;
                    }

                    // Recurrent kernel
                    res = new float[units, units];
                    for (int j = 0; j < units; j++)
                    {
                        for (int k = 0; k < units; k++)
                        {
                            res[j, k] = recurrentKernel[j, start + k];
                        }
                    }
                    switch (i)
                    {
                        case 0:
                            Ui = res;
                            break;
                        case 1:
                            Uf = res;
                            break;
                        case 2:
                            Uc = res;
                            break;
                        case 3:
                            Uo = res;
                            break;
                    }

                    // Biases
                    float[] resB = new float[units];
                    for (int k = 0; k < units; k++)
                    {
                        resB[k] = biases[start + k];
                    }
                    switch (i)
                    {
                        case 0:
                            bi = resB;
                            break;
                        case 1:
                            bf = resB;
                            break;
                        case 2:
                            bc = resB;
                            break;
                        case 3:
                            bo = resB;
                            break;
                    }
                }
            }

            private float[] RecurrentActivation(float[] x)
            {
                return ComputeActivations(x, ActivationFunctions.Sigmoid);
            }

            private float[] Activation(float[] x)
            {
                return ComputeActivations(x, func);
            }

            public override float[] FeedForward(float[] input)
            {
                float[] xi = AMath.Elementwise.Add(AMath.Matrix.Dot(input, Wi), bi);
                float[] xf = AMath.Elementwise.Add(AMath.Matrix.Dot(input, Wf), bf);
                float[] xc = AMath.Elementwise.Add(AMath.Matrix.Dot(input, Wc), bc);
                float[] xo = AMath.Elementwise.Add(AMath.Matrix.Dot(input, Wo), bo);

                float[] i = RecurrentActivation(AMath.Elementwise.Add(xi, AMath.Matrix.Dot(h, Ui)));
                float[] f = RecurrentActivation(AMath.Elementwise.Add(xf, AMath.Matrix.Dot(h, Uf)));
                float[] cNew = AMath.Elementwise.Add(AMath.Elementwise.Multiply(f, c), AMath.Elementwise.Multiply(i, Activation(AMath.Elementwise.Add(xc, AMath.Matrix.Dot(h, Uc)))));
                float[] o = RecurrentActivation(AMath.Elementwise.Add(xo, AMath.Matrix.Dot(h, Uo)));

                float[] hNew = AMath.Elementwise.Multiply(o, Activation(cNew));

                // Update internal state
                h = hNew;
                c = cNew;

                return hNew;
            }

            public override bool IsRecurrent()
            {
                return true;
            }
        }

        private static class ActivationFunctions
        {
            public static float Linear(float x)
            {
                return x;
            }

            public static float ReLU(float x)
            {
                return Math.Max(0.0f, x);
            }

            public static float Tanh(float x)
            {
                return (float)Math.Tanh(x);
            }

            public static float Sigmoid(float x)
            {
                return (float)(1.0 / (1.0 + Math.Exp(-x)));
            }

            public static float HardSigmoid(float x)
            {
                return Math.Max(0.0f, Math.Min(1.0f, x * 0.2f + 0.5f));
            }
        }
    }


}