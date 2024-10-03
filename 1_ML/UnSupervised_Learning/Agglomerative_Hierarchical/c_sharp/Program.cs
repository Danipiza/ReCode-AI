using System;
using System.Collections.Generic;

using ML_Utilities;
/*
<ItemGroup>    
    <Compile Include="../../../ml_utils.cs" />
</ItemGroup>
*/

/*
Agglomerative Hierarchical class.

Args:
    population (float[][]) : Population with all the individuals.
    C (int)                : Final number of clusters. 
                                used to finalize the execution.
    dist_cluster (int)     : Type of distance used between clusters.
    func (function)        : Euclidean or Manhattan distance.
*/
public class AgglomerativeHierarchical
{
    private List<double[]> population;
    private int n, d, C, dist_cluster; 
    
    private Func<double[], double[], double> func;
    
    // -------------------------------------------------------------------------
    public AgglomerativeHierarchical(
        List<double[]> population, 
        int C, int dist_cluster, 
        Func<double[], double[], double> func)
    {
        this.population=population;
        
        this.n=population.Count;
        this.d=population[0].Length;        
        this.C=C;
        this.dist_cluster=dist_cluster;

        this.func=func;
    }

    /*
    Execute function

    Return:
        assignment (int[][])  : Assignment of categories for each individual
                            in the self.C different stored number of clusters.
        centroids (float[][]) : Centroids for each individual
                            in the self.C different stored number of clusters.
    */
    public (int[][][], double[][][]) Execute()
    {
        if (dist_cluster==0) return CentroidExecution();
        else return ExecuteLink();
    }

    /*
    Execute centroid cluster distance

    Return:
        assignment (int[][])  : Assignment of categories for each individual
                            in the self.C different stored number of clusters.
        centroids (float[][]) : Centroids for each individual
                            in the self.C different stored number of clusters.
    */
    private (int[][][], double[][][]) CentroidExecution()
    {
        // -- 1st PHASE: Init Matrix -------------------------------------------
        double[][] M=new double[n-1][];
        for (int i=0;i<n-1;i++)
        {
            M[i]=new double[n];
            for (int j=i+1;j<n;j++)
            {
                M[i][j]=func(population[i], population[j]);
            }
        }

        // at the beginning, all the individuals are clusters
        var clusters=new List<List<int>>();
        for (int i=0;i<n;i++) clusters.Add(new List<int> {i});

        // Centroids of each cluster
        var centroid_clusters=new List<List<double>>();
        for (int i=0;i<n;i++)
        {
            var tmp=new List<double>();
            foreach(double x in population[i]) tmp.Add(x);
            centroid_clusters.Add(tmp);
        }
        

        // -- return variables -------------------------------------------------
        var assignment=new int[C][][];
        var centroids =new double[C][][];
        
        // -- 2nd PHASE: Main loop ---------------------------------------------
        // repeat the algorithm until there is only 'C' clusters
        
        int c1,c2;
        double distMin;
        for (int k=C;k<n;k++)
        {
            // -- search for the 2 closest clusters ----------------------------
            c1=-1; c2=-1;
            distMin=double.PositiveInfinity;

            for (int i=0;i<M.Length;i++)
            {
                for (int j=i+1;j<M[0].Length;j++)
                {
                    if (distMin>=M[i][j])
                    {
                        distMin=M[i][j];
                        c1=i;
                        c2=j;
                    }
                }
            }

            // -- join the 2 clusters ------------------------------------------
            foreach (int x in clusters[c2]) clusters[c1].Add(x);
            clusters.RemoveAt(c2);

            // -- update centroid ----------------------------------------------
            double[] tmp=new double[d];
            foreach (int x in clusters[c1])
            {
                for (int y=0;y<d;y++) tmp[y]+=population[x][y];
            }

            for (int x=0;x<d;x++) tmp[x]/=clusters[c1].Count;
            centroid_clusters[c1]=new List<double>(tmp);
            centroid_clusters.RemoveAt(c2);

            // -- delete row ---------------------------------------------------
            if (c2!=M.Length)
            {
                var MList=new List<double[]>(M);
                MList.RemoveAt(c2);
                M=MList.ToArray();
            }

            // -- delete column ------------------------------------------------            
            for(int i=0;i<M.Length;i++)
            {
                var rowList=new List<double>(M[i]);
                rowList.RemoveAt(c2);
                M[i]=rowList.ToArray();
            }



            // -- update column ------------------------------------------------
            int iCol=0;
            while (iCol != c1)
            {
                M[iCol][c1]=func(centroid_clusters[c1].ToArray(), 
                                 centroid_clusters[iCol].ToArray());
                iCol++;
            }

            // -- update row ---------------------------------------------------
            for (int iRow=c1+1;iRow<M[c1].Length;iRow++)
            {
                M[c1][iRow]=func(centroid_clusters[c1].ToArray(), 
                                 centroid_clusters[iRow].ToArray());
            }

            // -- Assign clusters and centroids if below 'C' clusters
            if (n-k-1<C)
            {
                assignment[n-k-1]=clusters.
                    Select(innerList => innerList.ToArray()).ToArray();
                centroids[n-k-1] =centroid_clusters.
                    Select(innerList => innerList.ToArray()).ToArray();
            }
        }

        return (assignment, centroids);
    }

    // Placeholder for the 'execute_link' function
    private (int[][][], double[][][]) ExecuteLink()
    {
        /*// -- 1st PHASE: Init Matrix -------------------------------------------
        double[][] M=new double[n-1][];
        for (int i=0;i<n-1;i++)
        {
            M[i]=new double[n];
            for (int j=i+1;j<n;j++)
            {
                M[i][j]=func(population[i], population[j]);
            }
        }

        // at the beginning, all the individuals are clusters
        var clusters=new List<List<int>>();
        for (int i=0;i<n;i++) clusters.Add(new List<int> {i});

        // Centroids of each cluster
        var centroid_clusters=new List<List<List<double>>>();
        for (int i=0;i<n;i++)
        {
            var tmp=new List<List<double>>();
            tmp.Add(population[i]);
            centroid_clusters.Add(tmp);
        }
        

        // -- return variables -------------------------------------------------
        var assignment=new int[C][][];
        var centroids =new double[C][][];
        
        // -- 2nd PHASE: Main loop ---------------------------------------------
        // repeat the algorithm until there is only 'C' clusters
        
        int c1,c2;
        double distMin;
        for (int k=C;k<n;k++)
        {
            // -- search for the 2 closest clusters ----------------------------
            c1=-1; c2=-1;
            distMin=double.PositiveInfinity;

            for (int i=0;i<M.Length;i++)
            {
                for (int j=i+1;j<M[0].Length;j++)
                {
                    if (distMin>=M[i][j])
                    {
                        distMin=M[i][j];
                        c1=i;
                        c2=j;
                    }
                }
            }

            // -- join the 2 clusters ------------------------------------------
            foreach (int x in clusters[c2]) 
            {
                clusters[c1].Add(x);
            }
            clusters.RemoveAt(c2);

            // -- update centroid ----------------------------------------------            
            foreach(List<List<double>> x in centroid_clusters[c2])
            {
                centroid_clusters[c1].Add(x);    
            }
            centroid_clusters.RemoveAt(c2);

            // -- delete row ---------------------------------------------------
            if (c2!=M.Length)
            {
                var MList=new List<double[]>(M);
                MList.RemoveAt(c2);
                M=MList.ToArray();
            }
            // -- delete column ------------------------------------------------            
            for(int i=0;i<M.Length;i++)
            {
                var rowList=new List<double>(M[i]);
                rowList.RemoveAt(c2);
                M[i]=rowList.ToArray();
            }

            
            // -- parameters ---------------------------------------------------            
            bool CompareSimple(double a, double b) => a>b;
            bool CompareFull  (double a, double b) => a<b;
            
            Func<double, double, bool> compare=CompareSimple;
            double limit=double.PositiveInfinity;

            if (this.dist_cluster==2)
            {
                compare=CompareFull;
                limit*=-1;
            }


            // -- update column ------------------------------------------------
            int iCol=0, c1N=centroid_clusters[c1].Count, c2N;
            double newDist, tmpDist;
            while (iCol!=c1)
            {
                newDist=limit;
                tmpDist=limit;

                c2N=len(centroid_clusters[i].Count);
                for(int a=0;i<c1N;a++)     // iterate through all 'c1' individuals
                { 
                    for(int b=0;i<c2N;b++) // iterate through all 'c2' individuals
                    {
                        tmpDist=func(centroid_clusters[c1][a],
                                     centroid_clusters[i][b]);
                        
                        //lower or greater distance
                        if(compare(newDist,tmpDist)) newDist=tmpDist;
                    }
                }
                M[i][c1]=newDist;             
                i+=1;
            }

            // -- update row ----------------------------------------------------
            for (int iRow=c1+1;iRow<M[c1].Length;iRow++)
            {
                M[c1][iRow]=func(centroid_clusters[c1].ToArray(), 
                                 centroid_clusters[iRow].ToArray());
            }

            // -- Assign clusters and centroids if below 'C' clusters
            if (n-k-1<C)
            {
                assignment[n-k-1]=clusters.
                    Select(innerList => innerList.ToArray()).ToArray();
                centroids[n-k-1] =centroid_clusters.
                    Select(innerList => innerList.ToArray()).ToArray();
            }
        }

        return (assignment, centroids);*/
        return (null,null);
    }
}

public static class Utils
{   

    /*
    Calculates de Euclidean distance of 
    two points in a 'd' dimensional space.

    Args:
        a (float[]) : Point in a d-dimensional space.
        b (float[]) : Point in a d-dimensional space.
    */
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

    /*
    Calculates the Manhattan distance of 
    two points in a 'd' dimensional space.

    Args:
        a (float[]) : Point in a d-dimensional space.
        b (float[]) : Point in a d-dimensional space.
    */
    public static double ManhattanDistance(double[] a, double[] b)
    {
        int n=a.Length;
        
        double sum=0;
        for (int i=0;i<n;i++) sum+=Math.abs(a[i]-b[i]);            
         
        return sum;
    }

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
                distance_cluster[i, j]=
                    Utils.EuclideanDistance(centroids[i], centroids[j]);
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
    Execute the algorithm. With the given parameters.

    Args:
        population (float[][]) : Population with all the individuals.
        C (int)                : Final number of clusters. 
                                    used to finalize the execution.
        dist_cluster (int)     : Type of distance used between clusters.
        func (function)        : Euclidean or Manhattan distance.
    */
    // execute
}

class Program
{
    static void Main(string[] args)
    {
        
        // -- Init population ------------------------------------------------------
        List<double[]> population=new List<double[]>();
        
        string filename="100_1_2D";
        population=Functions.ReadPopulation(filename,"clusters", 2);

        int C=7;

        // -- Individual distance --------------------------------------------------
        int distInd=0;
        string[] distIndName={ "Manhattan", "Euclidean" };
        
        
        Func<double[], double[], double> func=Utils.EuclideanDistance;
        if (distInd==0) func=Utils.ManhattanDistance;
        
        // -- Cluster distance -----------------------------------------------------
        int distCluster=0;
        
        string[] distClusterName={ "Centroid", "Simple Link", "Full Link" };
               
        

        AgglomerativeHierarchical AH = new AgglomerativeHierarchical(population, C, distCluster, func);
        AH.Execute();
    }
    
    
}
