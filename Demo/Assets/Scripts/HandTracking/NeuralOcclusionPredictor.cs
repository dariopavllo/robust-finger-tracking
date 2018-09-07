using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    /// <summary>
    /// Implements a marker predictor based on an autoencoder neural network topology.
    /// </summary>
    class NeuralOcclusionPredictor : OcclusionManager.Implementation
    {
        // Offset decay rate for re-entry discontinuities (Higher -> faster offset decay)
        private const float SmoothingConstant = 0.25f;

        // Maximum offset limit (in meters), to prevent instability
        private const float OffsetLimit = 0.1f;

        private readonly HandTemplate handTemplate;
        private readonly NeuralNetwork model;
        private readonly bool useOffset;
        private readonly bool smoothing;
        private Dictionary<string, Vector3> offsets;

        /// <summary>
        /// Initialize this predictor.
        /// </summary>
        /// <param name="modelFileName">The file name of the neural network model to import.</param>
        /// <param name="handTemplate">The hand template to use with this predictor.</param>
        /// <param name="useOffset">Use offset to correct occlusion discontinuities (default: true).</param>
        /// <param name="smoothing">Use offset to correct re-entry discontinuities (default: true).</param>
        public NeuralOcclusionPredictor(string modelFileName, HandTemplate handTemplate, bool useOffset = true, bool smoothing = true)
        {
            model = new NeuralNetwork(modelFileName);
            this.handTemplate = handTemplate;

            offsets = new Dictionary<string, Vector3>();
            foreach (var marker in handTemplate.MarkerList)
            {
                offsets[marker] = Vector3.zero;
            }

            this.useOffset = useOffset;
            this.smoothing = smoothing;
        }

        public Dictionary<string, Vector3> Predict(Dictionary<string, Vector3> availableMarkers, HashSet<string> occludedMarkers,
            Dictionary<string, Vector3> oldMarkers, HashSet<string> oldAvailableMarkers, float timeSinceLastFrame)
        {
            var availableMarkersSet = new HashSet<string>(availableMarkers.Keys);
            var markerList = handTemplate.MarkerList;

            double[,] R;
            double[] shift;

            // Put the hand in object space
            float[] input = ExtractInputFeatures(availableMarkers, out R, out shift);
            Dictionary<string, Vector3> output;
            if (occludedMarkers.Count > 0 || model.IsRecurrent())
            {
                // Predict using the neural network
                output = ExtractOutput(model.Predict(input));
            }
            else
            {
                // No need to predict using the neural network (there are no occlusions and the model is not recurrent)
                output = new Dictionary<string, Vector3>();
            }

            foreach (var marker in markerList)
            {
                if (availableMarkersSet.Contains(marker))
                {
                    output[marker] = Utils.RotateAndShift(availableMarkers[marker], R, shift);
                }
            }

            // If the set of occlusions has changed, the offsets must be recomputed
            if (!model.IsRecurrent() && oldMarkers.Keys.Count == markerList.Count)
            {
                if (!availableMarkersSet.SetEquals(oldAvailableMarkers))
                {
                    // Compute new offsets, using the data from the previous frame
                    var X = new Dictionary<string, Vector3>();
                    foreach (var marker in availableMarkersSet)
                    {
                        X[marker] = oldMarkers[marker];
                    }

                    double[,] RX;
                    double[] offsetX;
                    float[] inputX = ExtractInputFeatures(X, out RX, out offsetX);
                    var outputX = ExtractOutput(model.Predict(inputX));

                    var Y = new Dictionary<string, Vector3>();
                    foreach (var marker in markerList)
                    {
                        Y[marker] = oldMarkers[marker];
                    }

                    foreach (var marker in markerList)
                    {
                        if (availableMarkersSet.Contains(marker) && oldAvailableMarkers.Contains(marker))
                        {
                            // Do nothing
                        }
                        else if (availableMarkersSet.Contains(marker) && !oldAvailableMarkers.Contains(marker))
                        {
                            // Re-entry discontinuity
                            Vector3 y = Utils.RotateAndShift(Y[marker], R, shift);
                            offsets[marker] = y - output[marker];
                        }
                        else
                        {
                            // Occlusion discontinuity
                            Vector3 y = Utils.RotateAndShift(Y[marker], RX, offsetX);
                            offsets[marker] = y - outputX[marker];
                        }
                    }
                }

                foreach (var marker in markerList)
                {
                    if (smoothing)
                    {
                        // Decay offsets for re-entry discontinuities
                        if (availableMarkersSet.Contains(marker) && offsets[marker] != Vector3.zero)
                        {
                            float oldMagnitude = offsets[marker].sqrMagnitude;
                            offsets[marker] -= SmoothingConstant * timeSinceLastFrame * offsets[marker].normalized;
                            if (offsets[marker].sqrMagnitude >= oldMagnitude)
                            {
                                offsets[marker] = Vector3.zero;
                            }
                        }
                    }
                    else
                    {
                        if (availableMarkersSet.Contains(marker))
                        {
                            offsets[marker] = Vector3.zero;
                        }
                    }

                    // Add offsets
                    if (useOffset)
                    {
                        // Limit offset
                        if (offsets[marker].magnitude > OffsetLimit)
                        {
                            offsets[marker] = offsets[marker].normalized * OffsetLimit;
                        }
                        output[marker] += offsets[marker];
                    }
                }
            }

            // Move the hand back in world space
            handTemplate.InverseAlignTransform(output, R, shift);

            // This part is not really necessary, but it helps with improving numerical stability
            foreach (var marker in markerList)
            {
                if (availableMarkersSet.Contains(marker) && offsets[marker] == Vector3.zero)
                {
                    output[marker] = availableMarkers[marker];
                }
            }

            return output;
        }

        private float[] ExtractInputFeatures(Dictionary<string, Vector3> points, out double[,] R, out double[] offset)
        {
            points = new Dictionary<string, Vector3>(points); // Defensive copy
            handTemplate.AlignTransform(points, points, out R, out offset);

            var pointList = new List<Vector3>(handTemplate.MarkerList.Count);
            foreach (var marker in handTemplate.MarkerList)
            {
                if (points.ContainsKey(marker))
                {
                    pointList.Add(points[marker]);
                }
                else
                {
                    // Missing values are set to 0
                    pointList.Add(Vector3.zero);
                }
            }

            // Flatten input vectors
            float[] features = new float[pointList.Count * 3];
            for (int i = 0; i < pointList.Count; i++)
            {
                features[3 * i + 0] = pointList[i].x;
                features[3 * i + 1] = pointList[i].y;
                features[3 * i + 2] = pointList[i].z;
            }
            return features;
        }

        private Dictionary<string, Vector3> ExtractOutput(float[] output)
        {
            Debug.Assert(output.Length % 3 == 0, "Malformed neural network output");

            int numPoints = output.Length / 3;
            var markerList = handTemplate.MarkerList;
            Debug.Assert(numPoints == markerList.Count, "The neural network output does not match the hand template");

            // Reshape output
            var points = new Dictionary<string, Vector3>();
            for (int i = 0; i < numPoints; i++)
            {
                points[markerList[i]] = new Vector3(output[3 * i + 0], output[3 * i + 1], output[3 * i + 2]);
            }

            return points;
        }
    }

}