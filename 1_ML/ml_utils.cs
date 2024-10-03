using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;


namespace ML_Utilities
{
    public class Functions
    {
        /*
        Read a population of individuals with 'd' variables.

        The format of the file has to be as follows:
        [_,..,_], [_,..,_], ... ,[_,..,_]
        (All in 1 line.)

        Args:
            fileName (string)  : Name of the file that is going to be readed.
            algorithm (string) : Type of directory.
            d (int)            : Number of variables per individual.

        Return:
            ret (List<double[]>) : Individuals of the population.
        */
        public static List<double[]> ReadPopulation(string fileName, string algorithm,  int d) 
        {
            var ret=new List<double[]>();             // return list    

            try 
            {
                // get the root directory of the proyect
                string dir=Directory.GetCurrentDirectory();
                while (!dir.EndsWith("ReCode-AI")) 
                {
                    dir=Directory.GetParent(dir).FullName;
                }
                
                // name of the file
                if (fileName==null) fileName=Console.ReadLine();
                string path=Path.Combine(dir, ".Others", "files", 
                                        algorithm, fileName+".txt");
                
                // read the line 
                string data;
                data=File.ReadAllText(path); 

                
                // Remove brackets and split by "], ["
                string[] sets = data.Trim('[', ']').Split(new string[] { "], [" }, StringSplitOptions.RemoveEmptyEntries);

                // Parse each set of numbers
                foreach (string set in sets)
                {
                    // split by commas to get individual string values
                    string[] values=set.Split(',');

                    // Convert string values to double and store them in an array
                    double[] numbers = new double[values.Length];

                    for (int i = 0; i < values.Length; i++)
                    {
                        numbers[i] = double.Parse(values[i], CultureInfo.InvariantCulture);
                    }

                    // Add the array to the list
                    ret.Add(numbers);
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred: " + ex.Message);
            }
            
            

            

            return ret;
        }
    }
}