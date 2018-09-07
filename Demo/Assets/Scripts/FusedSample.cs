using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// Represents a sensor fusion frame.
/// This class contains the position and orientation of each hand joint (both in object space).
/// </summary>
public struct FusedSample : IComparable<FusedSample>
{
    public float time;

    public Vector3[] position;
    public Vector3[] orientation;
    public bool isValid;

    public FusedSample(float time, int numElements)
    {
        this.time = time;
        this.position = new Vector3[numElements];
        this.orientation = new Vector3[numElements];
        this.isValid = false;
    }

    public FusedSample(string input)
    {
        var strings = input.Split(',');
        int numElements = (strings.Length - 2) / 6;

        position = new Vector3[numElements];
        orientation = new Vector3[numElements];

        time = float.Parse(strings[0], CultureInfo.InvariantCulture);
        int i = 1;
        int element = 0;
        while (i < strings.Length - 1)
        {
            position[element].x = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            position[element].y = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            position[element].z = float.Parse(strings[i++], CultureInfo.InvariantCulture);

            orientation[element].x = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            orientation[element].y = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            orientation[element].z = float.Parse(strings[i++], CultureInfo.InvariantCulture);

            element++;
        }

        isValid = int.Parse(strings[i++]) == 1;
    }

    public static void Export(string csvPath, IEnumerable<FusedSample> samples, IEnumerable<string> jointNames)
    {
        using (var file = new StreamWriter(csvPath))
        {
            // Write header
            var features = new List<string>();
            features.Add("time");
            foreach (string name in jointNames)
            {
                features.Add(name + ".pos.x");
                features.Add(name + ".pos.y");
                features.Add(name + ".pos.z");
                features.Add(name + ".rot.x");
                features.Add(name + ".rot.y");
                features.Add(name + ".rot.z");
            }
            features.Add("isValid");
            file.WriteLine(string.Join(",", features.ToArray()));

            // Write records
            foreach (var s in samples)
            {
                file.WriteLine(s);
            }
        }
    }

    public static List<FusedSample> Import(string csvPath)
    {
        var output = new List<FusedSample>();
        using (var file = new StreamReader(csvPath))
        {
            string line;
            while ((line = file.ReadLine()) != null)
            {
                if (!line.StartsWith("time")) // Skip header
                {
                    output.Add(new FusedSample(line));
                }
                
            }
        }
        return output;
    }

    public override string ToString()
    {
        var strings = new List<string>();
        strings.Add(time.ToString(CultureInfo.InvariantCulture));
        for (int i = 0; i < position.Length; i++)
        {
            strings.Add(position[i].x.ToString(CultureInfo.InvariantCulture));
            strings.Add(position[i].y.ToString(CultureInfo.InvariantCulture));
            strings.Add(position[i].z.ToString(CultureInfo.InvariantCulture));

            strings.Add(orientation[i].x.ToString(CultureInfo.InvariantCulture));
            strings.Add(orientation[i].y.ToString(CultureInfo.InvariantCulture));
            strings.Add(orientation[i].z.ToString(CultureInfo.InvariantCulture));
        }

        strings.Add((isValid ? 1 : 0).ToString());
        return string.Join(",", strings.ToArray());
    }

    public int CompareTo(FusedSample other)
    {
        return time.CompareTo(other.time);
    }
}