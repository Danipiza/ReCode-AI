using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;

/*EXECUTE
dotnet build
dotnet run
*/

/*
K-Nearest Neighbors (KNN) algorithm.

Individual distances:
 - Manhattan
 - Euclidea

Once the algorithm finalize, an interface shows the assignment of:
    Initial population (at the left) 
    Final population (at the right)
*/

class Program
{
    
    /*
    Calculates de Euclidean distance of 
    two points in a 'd' dimensional space.

    Args:
        a (double[]) : Point in a d-dimensional space.
        b (double[]) : Point in a d-dimensional space.
    */
    public static double EuclideanDistance(double[] a, double[] b)
    {
        double ret=0;
        int d=a.Length;
        
        for (int i=0;i<d;i++) ret+=Math.Pow(a[i]-b[i], 2);        

        return Math.Sqrt(ret);
    }


    /*
    Calculates de Manhattan distance of 
    two points in a 'd' dimensional space.

    Args:
        a (double[]) : Point in a d-dimensional space.
        b (double[]) : Point in a d-dimensional space.
    */
    public static double ManhattanDistance(double[] a, double[] b) 
    {
        double ret=0;
        int d=a.Length;

        for (int i=0;i<d;i++) ret+=Math.Abs(a[i]-b[i]);
        
        return Math.Sqrt(ret);
    }


    /*
    Read a population of individuals with 'd' variables.

    The format of the file has to be as follows:
    [_,..,_], [_,..,_], ... ,[_,..,_]
    (All in 1 line.)

    Args:
        fileName (string) : Name of the file that is going to be readed.

    Return:
        ret (List<double[]>) : Individuals of the population.
    */
    public static List<double[]> ReadPopulation(string fileName) 
    {
        var ret=new List<double[]>();             // return list
        int d=int.Parse(fileName[^2].ToString()); // dimension
        var point=new double[d];                  // individual

        // get the root directory of the proyect
        string dir=Directory.GetCurrentDirectory();
        while (!dir.EndsWith("ReCode-AI")) 
        {
            dir=Directory.GetParent(dir).FullName;
        }
        
        // name of the file
        if (fileName==null) fileName=Console.ReadLine();
        string path=Path.Combine(dir, ".Others", "files", 
                                "clusters", fileName+".txt");
        
        // read the line 
        string data;
        try { data=File.ReadAllText(path); }
        catch (FileNotFoundException) 
        {
            Console.WriteLine($"The file '{fileName}.txt' doesn't exist.");
            return null;
        }

        // removes '[', ']'. And divides the file by ','
        data=data.Replace("[", "").Replace("]", "");
        var values=data.Split(", ").Select(double.Parse).ToList();

        // store the individuals in the return list    
        for (int i=0;i<values.Count;i+=d) 
        {
            point=new double[d];

            // read and store all the variables of the individual
            for (int j=0;j<d;j++) point[j]=values[i+j];
            ret.Add(point);
        }

        return ret;
    }

    /*
    Read an assignment of individuals.

    The format of the file has to be as follows:
    _ _ ... _
    (All in 1 line.)

    Args:
        fileName (string) : Name of the file that is going to be readed.

    Return:
        ret (int[]) : Categories of each individual.
    */
    public static List<int> ReadAssignment(string fileName) 
    {
        var ret=new List<int>(); // return list

        // get the root directory of the proyect
        string dir=Directory.GetCurrentDirectory();        
        while (!dir.EndsWith("ReCode-AI")) 
        {
            dir=Directory.GetParent(dir).FullName;
        }
          
        // name of the file
        if (fileName==null) fileName=Console.ReadLine();
        string path=Path.Combine(dir, ".Others", "files", 
                                "clusters", "assignment", fileName+".txt");

        // read the line
        string data;
        try { data=File.ReadAllText(path); }
        catch (FileNotFoundException) 
        {
            Console.WriteLine($"The file '{fileName}.txt' doesn't exist.");
            return null;
        }

        var values=data.Split(" ").Select(int.Parse).ToList();
        ret.AddRange(values);

        return ret;
    }

    /*
    PriorityQueue used to optimize the K nearest neighbors calculations.

    It is a maximum priority queue to store the nearest neighbors. 
    If a new individual is closer than the farest closest neighbor, 
        pops the top and push the new individual in the queue.

    Variables:
        priority = distance
        item     = index
    */
    public class MaxPriorityQueue 
    {
        
        private SortedList<double, int> queue=new SortedList<double, int>();

        public void Push(int item, double priority) 
        { 
            queue.Add(-priority, item); 
        }

        public double TopDistance() { return -queue.Keys.First(); }

        public int TopLabel() { return queue.Values.First(); }

        public int Pop() 
        {
            int item=queue.Values.First();
            queue.RemoveAt(0);
            return item;
        }

        public int Size() { return queue.Count; }
    }

    /*
    TODO

    */    
    /*public static void GUI(int numClusters, List<double[]> initPopulation, List<int> InitAssig, int n, 
                            List<double[]> population, List<int> assignment, int m) {
        var colors=new[] { Color.Blue, Color.Red, Color.Green, Color.Pink, Color.Yellow, Color.Magenta, 
                           Color.Brown, Color.DarkGreen, Color.Gray, Color.Fuchsia };

        
    }*/
    
    /*
    Classify one individual.

    Iterates through the given population and store in a priority queue
        the 'k' nearest neighbors and classify with the assignment
        of the stored individuals.

    Args:
        population (List<double[]>) : Categorized population.
        assignment (List<int>)      : Assignment of the population.
        ind (double[])              : Individual.
        numClusters (int)           : Number of possible clusters.
        k (int)                     : Number of neighbors.
        func (function)             : Manhattan or Euclidean distance.

    Return:
        ret (int) : Asignated cluster for the individual.
    */

    public static int ClassifyIndividual(
        List<double[]> population, List<int> assignment, double[] ind, 
        int numClusters, int k, Func<double[], double[], double> func) 
        {
        
        int n=population.Count;
        var pq=new MaxPriorityQueue();

        // calculate all distance an store the K nearest
        for (int i=0;i<n;i++) 
        {
            double distance=func(ind, population[i]);

            // space in queue?
            if (pq.Size()<k) pq.Push(assignment[i], distance);
            // if the actual distance is lower than the greater nearest distance
            // pops the greater and push the actual distance  
            else if (pq.TopDistance()>distance) 
            {
                pq.Pop();
                pq.Push(assignment[i], distance);
            }
        }

        // counts the number of neighbors for each cluster
        int[] labels=new int[numClusters];
        for (int i=0;i<k;i++) labels[pq.Pop()]++;

        // classify the individual with the 
        //   cluster with more occurrences
        return Array.IndexOf(labels, labels.Max());
    }

    /*
    Classify a population without updating.

    Iterates to classify each individual of the given population.

    Args:
        initPopulation (List<double[]>) : Categorized population.
        initAssig (List<int>)           : Assignment of the population.
        n (int)                         : Number of individuals in the categorized population
        population (List<double[]>)     : Population to categorize.
        n (int)                         : Number of individuals in the population to categorize
        numClusters (int)               : Number of possible clusters.
        k (int)                         : Number of neighbors.
        func (function)                 : Manhattan or Euclidean distance.
        
    Return:
        ret (int) : Asignated cluster for the individual.
    */
    public static void ExecuteNoUpdate(
        List<double[]> initPopulation, List<int> initAssig, int n, 
        List<double[]> population, int m, 
        int numClusters, int k, Func<double[], double[], double> func) 
        {

        var stopwatch=Stopwatch.StartNew();
        var assignment=new List<int>();

        for (int i=0;i<m;i++) 
        {
            assignment.Add(
                ClassifyIndividual(initPopulation, initAssig, 
                                    population[i], numClusters, k, func));
        }

        stopwatch.Stop();
        Console.WriteLine($"\nExecution time: {stopwatch.ElapsedMilliseconds}ms\n");

        //GUI(numClusters, initPopulation, initAssig, n, population, assignment, m);
    }

    /*
    Classify a population. 
    The population is updated through the iterations.

    Iterates to classify each individual of the given population.

    Args:
        initPopulation (List<double[]>) : Categorized population.
        initAssig (List<int>)           : Assignment of the population.
        n (int)                         : Number of individuals in the categorized population
        population (List<double[]>)     : Population to categorize.
        n (int)                         : Number of individuals in the population to categorize
        NumClusters (int)               : Number of possible clusters.
        k (int)                         : Number of neighbors.
        func (function)                 : Manhattan or Euclidean distance.
        
    Return:
        ret (int) : Asignated cluster for the individual.
    */
    public static void ExecuteUpdate(
        List<double[]> initPopulation, List<int> initAssig, int n, 
        List<double[]> population, int m, 
        int NumClusters, int k, Func<double[], double[], double> func) 
        {
        
        var stopwatch=Stopwatch.StartNew();
        var firstPopulation=new List<double[]>(initPopulation);
        var firstAssign=new List<int>(initAssig);
        var assignment=new List<int>();

        for (int x=0;x<m;x++) 
        {
            var pq=new MaxPriorityQueue();

            // calculate all distance an store the K nearest
            for (int i=0;i<n;i++) 
            {
                double distance=func(initPopulation[i], population[x]);

                // space in queue?
                if (pq.Size() < k) pq.Push(initAssig[i], distance);
                // if the actual distance is lower than the greater nearest distance
                // pops the greater and push the actual distance
                else if (pq.TopDistance()>distance) 
                {
                    pq.Pop();
                    pq.Push(initAssig[i], distance);
                }
            }

            // counts the number of neighbors for each cluster
            int[] labels=new int[numClusters];
            for (int i=0;i<k;i++) labels[pq.Pop()]++;
            

            // classify the individual with the 
            //   cluster with more occurrences
            //int ret=Array.IndexOf(labels, labels.Max());
            int ret=0;
            int maxOccur=labels[0];
            for(int i=1;i<numClusters;i++) 
            {
                if(maxOccur<labels[i])
                {
                    maxOccur=labels[i];
                    ret=i;
                }
            }
            
            // add the new assignment
            assignment.Add(ret);
            initAssig.Add(ret);

            // Update population
            initPopulation.Add(population[x]);
            n++;
        }

        stopwatch.Stop();
        Console.WriteLine($"Execution time: {stopwatch.ElapsedMilliseconds}ms");

        //GUI(numClusters, firstPopulation, firstAssign, n-m, population, assign, m);
    }

    static void Main() 
    {
        // 100_1_2D 7
        var initPopulation=ReadPopulation("1000_1_2D");
        var initAssig=ReadAssignment("1000_1_2D");
        int n=initPopulation.Count; 

        var population=ReadPopulation("100000_2D");
        population=population.Take(1000).ToList();
        int m=population.Count; 

        int numClusters=4; 
        int k=10; 
        int distance=1; 

        Func<double[], double[], double> func=EuclideanDistance;
        if (distance==0) func=ManhattanDistance;
        
        int update=0;

//------------------------------------------------------------------------------
        if (update==0) 
        {
            ExecuteNoUpdate(initPopulation, initAssig, n, 
                              population, m, numClusters, k, func);
        }
        else 
        {
            ExecuteUpdate(initPopulation, initAssig, n, 
                           population, m, numClusters, k, func);
        }
        
    }
}