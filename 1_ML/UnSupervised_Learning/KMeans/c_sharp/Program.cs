using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization; // Import this namespace for CultureInfo

using ML_Utilities;
/*
<ItemGroup>    
    <Compile Include="../../../ml_utils.cs" />
</ItemGroup>
*/


/*
Clustering algorithm, KMeans.
The objective is to categorize a population of data 
in a number of clusters

Args:
    k (int)                : Number of clusters.
    population (float[][]) : Population with all the individuals.    
    func (function)        : Euclidean or Manhattan distance.    
    print (bool)           : Boolean used to print information.
*/
public class KMeans
{
    private int k;
    private List<double[]> population;
    
    //           input1  , input2  , output
    private Func<double[], double[], double> func;

    
    private int d; // dimension of each individual
    private int n; // number of individuals

    private bool print;

    public KMeans(int k, List<double[]> population, 
        Func<double[], double[], double> func, bool print)
    {
        this.k=k;
        
        this.population =population;        
        this.d          =population[0].Length;        
        this.n          =population.Count;

        this.func=func;
        this.print=print;
        
        
        

        if (this.k>this.n)
        {
            Console.WriteLine("Error: Value of K is bigger than the size of the population");
        }
    }

    
    /*
    Compare two centroids.

    Args:
        a (float[]) : Centroid.
        b (float[]) : Centroid.
    */
    public bool CompareCentroids(List<double[]> a, List<double[]> b)
    {
        int n=a.Count;

        for (int i=0;i<n;i++)
        {
            for (int j=0;j<this.d;j++)
            {                
                if (a[i][j]!=b[i][j]) return false;
            }
        }

        return true;
    }

    /*
    Execute function.
    
    Return:
        assignment (int[])    : Last calculated assignment.
        centroids (float[][]) : Last calculated centroids.
    */
    public (int[], List<double[]>) Execute()
    {

        // -- Init -------------------------------------------------------------        
        // random initialization of the centroids 
        // pick self.k random individuals.
        Random random=new Random();

        Dictionary<int, bool> dic =new Dictionary<int, bool>();
        List<double[]> centroids  =new List<double[]>();

        for (int i=0;i<this.k;i++)
        {

            while (true)
            {
                int rand=random.Next(0, this.n);
                if (!dic.ContainsKey(rand))
                {
                    centroids.Add(this.population[rand]);
                    dic[rand]=true;
                    break;
                }
            }
        }

        int[] assignment=new int[this.n];
        for (int i=0;i<this.n;i++) assignment[i]=-1;
        int iterations=0;

        // while the centroids doesnt change
        while (true)
        {
            iterations++;

            // -- 1st PHASE: Assignment -----------------------------------------            
            for (int i=0;i<this.n;i++)
            {
                double dist=double.MaxValue;
                int cluster=-1;

                // compare the actual individual with each centroid          
                for (int j=0;j<this.k;j++)
                {
                    double tmp=this.func(this.population[i], centroids[j]);
                    if (tmp<dist)
                    {
                        dist=tmp;
                        cluster=j;
                    }
                }

                // assign the closest cluster to the actual individual (i-th) 
                assignment[i]=cluster;
            }

            // -- 2nd PHASE: Update centroids -----------------------------------
            // number of individuals assigned for each cluster
            int[] cluster_size=new int[this.k];
            List<double[]> new_centroids=new List<double[]>();
            
            for (int i=0;i<this.k;i++) new_centroids.Add(new double[this.d]);
            

            for (int i=0;i<this.n;i++)
            {
                for (int j=0;j<this.d;j++)
                {
                    new_centroids[assignment[i]][j]+=this.population[i][j];
                }
                cluster_size[assignment[i]]++;
            }

            if (this.print)
            {
                Console.WriteLine("Population: "+
                    string.Join(", ", this.population));
                Console.WriteLine("Centroids: "+
                    string.Join(", ", centroids));
                Console.WriteLine("Cluster sizes: "+
                    string.Join(", ", cluster_size));
            }

            // calculate new centroids. for each cluster divide by the it size
            for (int i=0;i<this.k;i++)
            {
                for (int j=0;j<this.d;j++)
                {
                    if (cluster_size[i]!=0) 
                    {
                        new_centroids[i][j]/=cluster_size[i];
                    }
                }
            }

            if (this.print)
            {
                Console.WriteLine("New centroids: "+ 
                    string.Join(", ", new_centroids));
            }

            // -- 3rd PHASE: Compare centroids ---------------------------------
            if (CompareCentroids(centroids, new_centroids)) break;
            centroids=new_centroids; // dont finalizes. update centroids
        }

        return (assignment, centroids);
    }
}

// euclidean distance 
public static class Utils
{
    public static double EuclideanDistance(double[] a, double[] b)
    {
        int n=a.Length;
        
        double sum=0;
        for (int i=0;i<n;i++)
        {
            sum+=Math.Pow(a[i]-b[i], 2);
        }
         
        return Math.Sqrt(sum);
    }

    
}

public static class KMeansUtils
{
    /*
    Davies-Bouldin index.

    DB=(1/k)*Sum(i=1&&i!=j -> k)[max((avg_distance(i,centroid[i])+
                                    avg_distance(i,centroid[i])/
                                    (centroids_distance(i,j)))]

    Args:
        population (float[][]) : Population with all the individuals.
        assignment (int[])     : Assignment of categories for each individual.
        k (int)                : Number of clusters.
        centroids (float[][])  : Centroids of each cluster.
    */
    public static double DaviesBouldin(List<double[]> population, int k, 
        int[] assignment, List<double[]> centroids)
    {
        double ret=0.0;
        int n=population.Count;
        double[] avg_distance=new double[k];
        int[] cluster_size=new int[k];

        // calculate average distances
        for (int i=0;i<n;i++)
        {
            avg_distance[assignment[i]]+=Utils.EuclideanDistance(
                centroids[assignment[i]], population[i]);
            
            cluster_size[assignment[i]]++;
        }

        for (int i=0;i<k;i++)
        {
            if (cluster_size[i]!=0) avg_distance[i]/=cluster_size[i];
        }

        // calculate distances between centroids
        double[,] distance_cluster=new double[k, k];
        for (int i=0;i<k-1;i++)
        {
            for (int j=i+1;j<k;j++)
            {
                distance_cluster[i, j]=Utils.EuclideanDistance(centroids[i], centroids[j]);
                distance_cluster[j, i]=distance_cluster[i, j];
            }
        }

        // calculate DB Index
        for (int i=0;i<k;i++)
        {
            double maxVal=0.0;
            for (int j=0;j<k;j++)
            {
                if (i==j) continue;

                double tmp=(avg_distance[i]+avg_distance[j])/
                    distance_cluster[i, j];
                if (tmp>maxVal) maxVal=tmp;
            }
            ret+=maxVal;
        }

        return ret/k;
    }

    
    /*
    The evaluation function calculates the cuadratic sum of the distances 
    of each individual with its cluster centroid. Euclidean distance.

    Args:
        population (float[][]) : Population with all the individuals.
        assignment (int[])     : Assignment of categories for each individual.
        centroids (float[][])  : Centroids of each cluster.
    */
    public static double Evaluation(List<double[]> population, int[] assignment, List<double[]> centroids)
    {
        int n=population.Count;

        double ret=0.0;        
        for (int i=0;i<n;i++)
        {
            ret+=Utils.EuclideanDistance(population[i], centroids[assignment[i]]);
        }

        return ret;
    }

    /*
    Execute the algorithm.

    Args:
        population (float[][]) : Population with all the individuals.
        k (int)                : Number of clusters.
        func (function)        : Euclidean or Manhattan distance.
    */
    public static (int[], List<double[]>) ExecuteAlgorithm(
        List<double[]> population, int k, 
        Func<double[], double[], double> func)
    {
        KMeans km=new KMeans(k, population, func, false);

        var (assignment, centroids)=km.Execute();
        double eval=Evaluation(population, assignment, centroids);
        Console.WriteLine($"Obtained evaluation: {eval}");

        return (assignment, centroids);
    }

    /*
    Execute a depth search with different numbers of K.

    Args:
        population (float[][]) : Population with all the individuals.
        max_clust (int)        : Maximum number of clusters.
        times (int)            : Number of times executed each K.     
        func (function)        : Euclidean or Manhattan distance.
    */
    public static void ExecuteSearch(List<double[]> population, int maxClust, int times, Func<double[], double[], double> func)
    {
        List<double> fits=new List<double>();
        List<int[]> bests=new List<int[]>();
        List<List<double[]>> bestsCentroids=new List<List<double[]>>();

        for (int k=1;k<=maxClust;k++)
        {
            double bestEval=double.MaxValue;
            int[] bestAssignment=null;
            List<double[]> bestCentroid=null;

            for (int t=0;t<times;t++)
            {
                var (assignment, centroids)=ExecuteAlgorithm(population, k, func);
                double eval=Evaluation(population, assignment, centroids);
                if (eval<bestEval)
                {
                    bestEval=eval;
                    bestAssignment=assignment;
                    bestCentroid=centroids;
                }
            }

            fits.Add(bestEval);
            bests.Add(bestAssignment);
            bestsCentroids.Add(bestCentroid);

            Console.WriteLine($"K: {k}\tBest eval: {bestEval}");
        }

        // Davies-Bouldin calculations        
        int index=0;
        double db, bestDB=double.MaxValue;
        for (int i=2;i<=maxClust;i++)
        {
            db=DaviesBouldin(population, i, bests[i-1], bestsCentroids[i-1]);
            if(bestDB>db)   
            {
                bestDB=db;
                index=i;
            }
            Console.WriteLine($"Davies Bouldin index (K={i}): {db}");            
        }

         
        Console.WriteLine($"Best number of clusters: {index}");
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Replace this with real data population
        List<double[]> population=new List<double[]>();

        // Example data loading function would be needed (not provided in original code)
        string filename="100_1_2D";
        population = Functions.ReadPopulation(filename,"clusters", 2);

        int k=4;

        // Example using Euclidean distance function
        Func<double[], double[], double> func=Utils.EuclideanDistance;
        KMeansUtils.ExecuteSearch(population, k, 8, func);
        //KMeansUtils.ExecuteAlgorithm(population, 4, func);
    }
}