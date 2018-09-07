using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    class NaiveOcclusionPredictor : OcclusionManager.Implementation
    {
        public Dictionary<string, Vector3> Predict(Dictionary<string, Vector3> availableMarkers, HashSet<string> occludedMarkers,
            Dictionary<string, Vector3> oldMarkers, HashSet<string> oldAvailableMarkers, float timeSinceLastFrame)
        {
            var output = new Dictionary<string, Vector3>();
            foreach (var marker in occludedMarkers)
            {
                if (oldMarkers.ContainsKey(marker))
                {
                    output[marker] = oldMarkers[marker];
                }
            }
            return output;
        }
    }

}