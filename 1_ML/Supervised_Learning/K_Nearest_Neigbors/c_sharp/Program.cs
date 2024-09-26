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

class Program{
    
    /*
    Calculates de Euclidean distance of 
    two points in a 'd' dimensional space.

    Args:
        a (double[]) : Point in a d-dimensional space.
        b (double[]) : Point in a d-dimensional space.
    */
    public static double euclidean_distance(double[] a, double[] b){
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
    public static double manhattan_distance(double[] a, double[] b) {
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
        file_name (string) : Name of the file that is going to be readed.

    Return:
        ret (List<double[]>) : Individuals of the population.
    */
    public static List<double[]> read_population(string fileName) {
        var ret=new List<double[]>();             // return list
        int d=int.Parse(fileName[^2].ToString()); // dimension
        var point=new double[d];                  // individual

        // get the root directory of the proyect
        string dir = Directory.GetCurrentDirectory();
        while (!dir.EndsWith("ReCode-AI")) dir=Directory.GetParent(dir).FullName;
        
        // name of the file
        if (fileName==null) fileName=Console.ReadLine();
        string path=Path.Combine(dir, ".Others", "files", "clusters", fileName+".txt");
        
        // read the line 
        string data;
        try { data=File.ReadAllText(path); }
        catch (FileNotFoundException) {
            Console.WriteLine($"The file '{fileName}.txt' doesn't exist.");
            return null;
        }

        // removes '[', ']'. And divides the file by ','
        data=data.Replace("[", "").Replace("]", "");
        var values=data.Split(", ").Select(double.Parse).ToList();

        // store the individuals in the return list    
        for (int i=0;i<values.Count;i+=d) {
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
        file_name (string) : Name of the file that is going to be readed.

    Return:
        ret (int[]) : Categories of each individual.
    */
    public static List<int> read_assignment(string fileName) {
        var ret=new List<int>(); // return list

        // get the root directory of the proyect
        string dir=Directory.GetCurrentDirectory();        
        while (!dir.EndsWith("ReCode-AI")) dir=Directory.GetParent(dir).FullName;

        // name of the file
        if (fileName==null) fileName=Console.ReadLine();
        string path=Path.Combine(dir, ".Others", "files", "clusters", "assignment", fileName+".txt");

        // read the line
        string data;
        try { data=File.ReadAllText(path); }
        catch (FileNotFoundException) {
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
    public class MaxPriorityQueue {
        private SortedList<double, int> _queue = new SortedList<double, int>();

        public void Push(int item, double priority) { 
            _queue.Add(-priority, item); 
        }

        public double TopDistance() { return -_queue.Keys.First(); }

        public int TopLabel() { return _queue.Values.First(); }

        public int Pop() {
            int item = _queue.Values.First();
            _queue.RemoveAt(0);
            return item;
        }

        public int Size() { return _queue.Count; }
    }

    /*
    TODO

    */    
    /*public static void GUI(int num_clusters, List<double[]> init_population, List<int> init_assig, int n, 
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
        num_clusters (int)          : Number of possible clusters.
        k (int)                     : Number of neighbors.
        func (function)             : Manhattan or Euclidean distance.

    Return:
        ret (int) : Asignated cluster for the individual.
    */
    public static int classify_individual(List<double[]> population, List<int> assignment, double[] ind, 
        int num_clusters, int k, Func<double[], double[], double> func) {
        
        int n=population.Count;
        var pq=new MaxPriorityQueue();

        // calculate all distance an store the K nearest
        for (int i=0;i<n;i++) {
            double distance=func(ind, population[i]);

            // space in queue?
            if (pq.Size()<k) pq.Push(assignment[i], distance);
            // if the actual distance is lower than the greater nearest distance
            // pops the greater and push the actual distance  
            else if (pq.TopDistance()>distance) {
                pq.Pop();
                pq.Push(assignment[i], distance);
            }
        }

        // counts the number of neighbors for each cluster
        int[] labels=new int[num_clusters];
        for (int i=0;i<k;i++) labels[pq.Pop()]++;

        // classify the individual with the 
        //   cluster with more occurrences
        return Array.IndexOf(labels, labels.Max());
    }

    /*
    Classify a population without updating.

    Iterates to classify each individual of the given population.

    Args:
        init_population (List<double[]>) : Categorized population.
        init_assig (List<int>)           : assignment of the population.
        n (int)                          : Number of individuals in the categorized population
        population (List<double[]>)      : Population to categorize.
        n (int)                          : Number of individuals in the population to categorize
        num_clusters (int)               : Number of possible clusters.
        k (int)                          : Number of neighbors.
        func (function)                  : Manhattan or Euclidean distance.
        
    Return:
        ret (int) : Asignated cluster for the individual.
    */
    public static void execute_no_update(List<double[]> init_population, List<int> init_assig, int n, 
        List<double[]> population, int m, 
        int num_clusters, int k, Func<double[], double[], double> func) {

        var stopwatch=Stopwatch.StartNew();
        var assignment=new List<int>();

        for (int i=0;i<m;i++) {
            assignment.Add(classify_individual(init_population, init_assig, population[i], num_clusters, k, func));
        }

        stopwatch.Stop();
        Console.WriteLine($"\nExecution time: {stopwatch.ElapsedMilliseconds}ms\n");

        //GUI(num_clusters, init_population, init_assig, n, population, assignment, m);
    }

    /*
    Classify a population. 
    The population is updated through the iterations.

    Iterates to classify each individual of the given population.

    Args:
        init_population (List<double[]>) : Categorized population.
        init_assig (List<int>)           : assignment of the population.
        n (int)                          : Number of individuals in the categorized population
        population (List<double[]>)      : Population to categorize.
        n (int)                          : Number of individuals in the population to categorize
        num_clusters (int)               : Number of possible clusters.
        k (int)                          : Number of neighbors.
        func (function)                  : Manhattan or Euclidean distance.
        
    Return:
        ret (int) : Asignated cluster for the individual.
    */
    public static void execute_update(List<double[]> init_population, List<int> init_assig, int n, 
        List<double[]> population, int m, 
        int num_clusters, int k, Func<double[], double[], double> func) {
        
        var stopwatch=Stopwatch.StartNew();
        var firstPopulation=new List<double[]>(init_population);
        var firstAssign=new List<int>(init_assig);
        var assignment=new List<int>();

        for (int x=0;x<m;x++) {
            var pq=new MaxPriorityQueue();

            // calculate all distance an store the K nearest
            for (int i=0;i<n;i++) {
                double distance=func(init_population[i], population[x]);

                // space in queue?
                if (pq.Size() < k) pq.Push(init_assig[i], distance);
                // if the actual distance is lower than the greater nearest distance
                // pops the greater and push the actual distance
                else if (pq.TopDistance()>distance) {
                    pq.Pop();
                    pq.Push(init_assig[i], distance);
                }
            }

            // counts the number of neighbors for each cluster
            int[] labels=new int[num_clusters];
            for (int i=0;i<k;i++) labels[pq.Pop()]++;
            

            // classify the individual with the 
            //   cluster with more occurrences
            //int ret=Array.IndexOf(labels, labels.Max());
            int ret=0;
            int max_occur=labels[0];
            for(int i=1;i<num_clusters;i++) {
                if(max_occur<labels[i]){
                    max_occur=labels[i];
                    ret=i;
                }
            }
            
            // add the new assignment
            assignment.Add(ret);
            init_assig.Add(ret);

            // Update population
            init_population.Add(population[x]);
            n++;
        }

        stopwatch.Stop();
        Console.WriteLine($"Execution time: {stopwatch.ElapsedMilliseconds}ms");

        //GUI(num_clusters, firstPopulation, firstAssign, n-m, population, assign, m);
    }

    static void Main() {
        // 100_1_2D 7
        var init_population=read_population("1000_1_2D");
        var init_assig=read_assignment("1000_1_2D");
        int n=init_population.Count; 

        var population=read_population("100000_2D");
        population=population.Take(1000).ToList();
        int m=population.Count; 

        int num_clusters=4; 
        int k=10; 
        int distance=1; 

        Func<double[], double[], double> func = euclidean_distance;
        if (distance==0) func=manhattan_distance;
        
        int update=0;

        
        if (update==0) execute_no_update(init_population, init_assig, n, population, m, num_clusters, k, func);
        else execute_update(init_population, init_assig, n, population, m, num_clusters, k, func);
        
    }
}