using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    /// <summary>
    /// Implements a linear occlusion predictor that expresses occluded markers as an affine combination of available markers.
    /// For a detailed description of this approach, see Section 3.2.2 of the report.
    /// </summary>
    class LinearOcclusionPredictor : OcclusionManager.Implementation
    {
        private double[,] weights;

        public Dictionary<string, Vector3> Predict(Dictionary<string, Vector3> availableMarkers, HashSet<string> occludedMarkers,
            Dictionary<string, Vector3> oldMarkers, HashSet<string> oldAvailableMarkers, float timeSinceLastFrame)
        {
            if (!oldAvailableMarkers.SetEquals(availableMarkers.Keys))
            {
                var X = new List<Vector3>();
                foreach (var idx in availableMarkers.Keys)
                {
                    X.Add(oldMarkers[idx]);
                }

                var Y = new List<Vector3>();
                foreach (var idx in occludedMarkers)
                {
                    Y.Add(oldMarkers[idx]);
                }
                weights = Utils.ComputeImputationWeights(X, Y);
            }

            var output = new Dictionary<string, Vector3>();
            if (occludedMarkers.Count > 0 && weights != null)
            {
                var Xp = new List<Vector3>();
                foreach (var idx in availableMarkers.Keys)
                {
                    Xp.Add(availableMarkers[idx]);
                }
                Vector3[] predicted = Utils.Impute(Xp, weights);
                for (int i = 0; i < predicted.Length; i++)
                {
                    output[occludedMarkers.ElementAt(i)] = predicted[i];
                }

            }
            return output;
        }
    }

}