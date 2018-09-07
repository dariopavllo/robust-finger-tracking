using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    /// <summary>
    /// This class implements an occlusion manager, which allows the user to stack multiple implementations
    /// in a pipeline-based system.
    /// </summary>
    class OcclusionManager
    {
        private readonly HashSet<string> allMarkers;
        private Dictionary<string, Vector3> oldMarkers;
        private HashSet<string> oldAvailableMarkers;
        private List<Implementation> pipeline;

        /// <summary>
        /// Initialize this occlusion manager.
        /// </summary>
        /// <param name="allMarkers">The set of all markers (the order is not important).</param>
        public OcclusionManager(IEnumerable<string> allMarkers)
        {
            this.allMarkers = new HashSet<string>(allMarkers);
            pipeline = new List<Implementation>();
            oldAvailableMarkers = this.allMarkers;
        }

        /// <summary>
        /// Predict the positions of occluded markers.
        /// </summary>
        /// <param name="markerIds">An array with the marker names.</param>
        /// <param name="markerPositions">An array with the marker positions (where NaN = missing).</param>
        /// <param name="timeSinceLastFrame">The time passed since the last frame.</param>
        public void Update(string[] markerIds, Vector3[] markerPositions, float timeSinceLastFrame)
        {
            Debug.Assert(markerIds.Length == markerPositions.Length, "Array size mismatch");
            var markers = new Dictionary<string, Vector3>();
            for (int i = 0; i < markerIds.Length; i++)
            {
                markers[markerIds[i]] = markerPositions[i];
            }

            Update(markers, timeSinceLastFrame);

            // Save result to input array
            for (int i = 0; i < markerIds.Length; i++)
            {
                if (markers.ContainsKey(markerIds[i]))
                {
                    markerPositions[i] = markers[markerIds[i]];
                }
            }
        }

        /// <summary>
        /// Predict the positions of occluded markers.
        /// </summary>
        /// <param name="markerPositions">The positions of available markers (no NaNs allowed).</param>
        /// <param name="timeSinceLastFrame">The time passed since the last frame.</param>
        public void Update(Dictionary<string, Vector3> markerPositions, float timeSinceLastFrame)
        {
            var availableMarkers = new Dictionary<string, Vector3>();
            foreach (var marker in markerPositions)
            {
                if (allMarkers.Contains(marker.Key) && !float.IsNaN(marker.Value.x))
                {
                    availableMarkers[marker.Key] = marker.Value;
                }
            }

            // The first frame is discarded to initialize the internal state of the predictor.
            if (oldMarkers == null)
            {
                oldMarkers = new Dictionary<string, Vector3>();
                foreach (var marker in availableMarkers)
                {
                    oldMarkers[marker.Key] = marker.Value;
                }
                return;
            }

            // Run the prediction pipeline sequentially. The points that are not predicted by the
            // first stage will be predicted by the second stage, and so on.
            var availableMarkersSet = new HashSet<string>(availableMarkers.Keys);
            foreach (var impl in pipeline)
            {
                var occludedMarkers = new HashSet<string>(allMarkers.Except(availableMarkers.Keys));
                var result = impl.Predict(availableMarkers, occludedMarkers, oldMarkers, oldAvailableMarkers, timeSinceLastFrame);
                foreach (var marker in result)
                {
                    availableMarkers[marker.Key] = marker.Value;
                }
            }

            foreach (var marker in availableMarkers)
            {
                markerPositions[marker.Key] = marker.Value;
                oldMarkers[marker.Key] = marker.Value;
            }

            oldAvailableMarkers = availableMarkersSet;
        }

        /// <summary>
        /// Add a predictor to the end of the pipeline.
        /// </summary>
        /// <param name="impl">The implementation to add</param>
        public void AddImplementation(Implementation impl)
        {
            pipeline.Add(impl);
        }

        /// <summary>
        /// Remove a predictor from this pipeline.
        /// </summary>
        /// <param name="impl">The implementation to remove</param>
        public void RemoveImplementation(Implementation impl)
        {
            pipeline.Remove(impl);
        }

        /// <summary>
        /// Remove all implementations from this pipeline.
        /// </summary>
        public void RemoveAllImplementations()
        {
            pipeline.Clear();
        }

        /// <summary>
        /// This interface must be implemented by all concrete implementations.
        /// </summary>
        public interface Implementation
        {
            /// <summary>
            /// Predict the position of occluded markers.
            /// </summary>
            /// <param name="availableMarkers">The positions of available markers at this frame.</param>
            /// <param name="occludedMarkers">The set of occluded markers at this frame.</param>
            /// <param name="oldMarkers">The positions of ALL markers (whether real or predicted) at the previous frame.</param>
            /// <param name="oldAvailableMarkers">The set of available markers at the previous frame.</param>
            /// <param name="timeSinceLastFrame">The time passed since the last frame.</param>
            /// <returns></returns>
            Dictionary<string, Vector3> Predict(Dictionary<string, Vector3> availableMarkers, HashSet<string> occludedMarkers,
                Dictionary<string, Vector3> oldMarkers, HashSet<string> oldAvailableMarkers, float timeSinceLastFrame);
        }

    }

}