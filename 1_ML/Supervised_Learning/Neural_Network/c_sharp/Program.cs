using System;
using System.Collections.Generic;
using System.Diagnostics;

public class NeuralNetwork
{
    private int inputSize;
    private List<int> hiddenSize;
    private int outputSize;
    private List<List<List<double>>> weights;
    private double lr;
    private List<int> layers;
    private List<List<double>> output;

    public NeuralNetwork(int inputSize, List<int> hiddenSize, int outputSize, 
                         List<List<List<double>>> weights, double lr)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.lr = lr;
        layers = new List<int> { inputSize };
        layers.AddRange(hiddenSize);
        layers.Add(outputSize);

        // random init
        if (weights == null)
        {
            this.weights = new List<List<List<double>>>();
            Random rand = new Random();
            
            for (int i = 0; i < layers.Count - 1; i++)
            {
                List<List<double>> layerWeights = new List<List<double>>();
                for (int j = 0; j < layers[i]; j++)
                {
                    List<double> neuronWeights = new List<double>();
                    for (int k = 0; k < layers[i + 1]; k++)
                    {
                        neuronWeights.Add(rand.NextDouble() * 2 - 1);
                    }
                    layerWeights.Add(neuronWeights);
                }
                this.weights.Add(layerWeights);
            }
        }
        else
        {
            this.weights = weights;
        }
    }

    // Activation function
    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    // Derivative of sigmoid function
    private double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    // Forward propagation
    public List<double> Forward(List<double> x)
    {
        output = new List<List<double>> { x };

        for (int i = 0; i < layers.Count - 1; i++)
        {
            List<double> inputsLayer = output[output.Count - 1];
            List<double> outputsLayer = new List<double>(new double[layers[i + 1]]);

            for (int j = 0; j < layers[i + 1]; j++)
            {
                double sumV = 0;
                for (int k = 0; k < layers[i]; k++)
                {
                    sumV += inputsLayer[k] * weights[i][k][j];
                }
                outputsLayer[j] = Sigmoid(sumV); // Activation function
            }

            output.Add(outputsLayer);
        }

        // Return the prediction (the last layer output)
        return output[output.Count - 1];
    }

    // Backpropagation method
    public void Backpropagation(List<double> x, List<double> label)
    {
        Forward(x);

        List<double> errors = new List<double>();
        for (int i = 0; i < outputSize; i++)
        {
            errors.Add((label[i] - output[output.Count - 1][i]) * 
                       SigmoidDerivative(output[output.Count - 1][i]));
        }

        // Iterate through the layers in reverse
        for (int i = layers.Count - 2; i >= 0; i--)
        {
            List<double> newErrors = new List<double>(new double[layers[i]]);

            for (int j = 0; j < layers[i]; j++)
            {
                double sumV = 0;
                for (int k = 0; k < layers[i + 1]; k++)
                {
                    sumV += errors[k] * weights[i][j][k];
                }

                newErrors[j] = sumV * SigmoidDerivative(output[i][j]);

                // Update the weights
                for (int k = 0; k < layers[i + 1]; k++)
                {
                    weights[i][j][k] += lr * errors[k] * output[i][j];
                }
            }

            errors = newErrors;
        }
    }

    // Calculate the minimum value in a list and its index
    public static (double, int) MinimumValue(List<double> values)
    {
        double minVal = double.PositiveInfinity;
        int index = -1;

        for (int i = 0; i < values.Count; i++)
        {
            if (values[i] < minVal)
            {
                minVal = values[i];
                index = i;
            }
        }

        return (minVal, index);
    }
}



public class Program
{

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
    public static List<double[]> ReadPopulation(string fileName, string algorithm, int d) 
    {
        var ret=new List<double[]>();             // return list        
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
                                algorithm, fileName+".txt");
        
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

    // Training method
    public static List<double> TrainingMethod(List<double[]> dataset, List<double[]> evalDataset,
                                              NeuralNetwork nn, int numEpochs, bool print = false)
    {
        List<double> ret = new List<double>();

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            foreach (var data in dataset)
            {
                List<double> input = new List<double>();
                input.Add(data[0]);
                input.Add(data[1]);
                List<double> label = new List<double> { data[2] };
                nn.Backpropagation(input, label);
            }

            double error = 0;
            foreach (var data in evalDataset)
            {
                List<double> prediction = new List<double>();
                prediction.Add(data[0]);
                prediction.Add(data[1]);
                error += Math.Abs(prediction[0] - data[2]);
                Console.WriteLine($"Real: {data[2]} \tPred: {prediction[0]}");        
            }

            ret.Add(error);
            if (print) Console.WriteLine($"Epoch {epoch} - Total error = {error}");
        }

        return ret;
    }

    // Execute the training with a fixed learning rate
    public static void Execute_a(List<double[]> dataset, List<double[]> evalDataset, int numEpochs, double lr,
                               int inputSize, List<int> hiddenSize, int outputSize, List<List<List<double>>> model)
    {
        
        var stopwatch = Stopwatch.StartNew();        
        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize, model, lr);
        List<double> error = TrainingMethod(dataset, evalDataset, nn, numEpochs);

        stopwatch.Stop();
        Console.WriteLine($"Final error: {error[error.Count - 1]}");
        Console.WriteLine($"Execution time: {stopwatch.Elapsed.TotalSeconds}s");
    }

    // Execute search for the optimal learning rate
    public static void ExecuteSearch(List<double[]> dataset, List<double[]> evalDataset, int numEpochs,
                                     int inputSize, List<int> hiddenSize, int outputSize, List<List<List<double>>> model)
    {
        Console.WriteLine($"Hidden layer sizes: {string.Join(", ", hiddenSize)}\tNumber of epochs: {numEpochs}");

        var stopwatch = Stopwatch.StartNew();
        List<double> learningRates = new List<double>();
        for (int i = 1; i <= 20; i++) learningRates.Add(0.01 * i);

        List<double> errors = new List<double>();
        List<int> epochs = new List<int>();

        foreach (var lr in learningRates)
        {
            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize, model, lr);
            List<double> lrError = TrainingMethod(dataset, evalDataset, nn, numEpochs);

            (double err, int epoch) = NeuralNetwork.MinimumValue(lrError);
            errors.Add(err);
            epochs.Add(epoch);
        }

        stopwatch.Stop();
        Console.WriteLine($"Execution time: {stopwatch.Elapsed.TotalSeconds}s");

        // Optionally, GUI function can be implemented for plotting if needed.
        // Example: PlotGUI(learningRates, errors, epochs); // For visualization
    }

    public static void Main(string[] args)
    {
        
        // Example dataset and evaluation dataset
        List<double[]> dataset=ReadPopulation("population_80","neural_network",3);
        List<double[]> evalDataset=new List<double[]>();
        for(int i=0;i<10;i++) evalDataset.Add(dataset[i]);

        foreach(double[] d in dataset)
        {
           Console.WriteLine($"Height: {d[0]}\tWeight: {d[1]}\tBMI: {d[2]}"); 
        }
        
        // Neural Network configuration
        int inputSize = 2;
        List<int> hiddenSize = new List<int> { 10 };
        int outputSize = 1;
        List<List<List<double>>> model = null;
        int numEpochs = 10;
        double lr = 0.001;

        // Execute the search for the optimal learning rate
        //ExecuteSearch(dataset, evalDataset, numEpochs, inputSize, hiddenSize, outputSize, model);
        Execute_a(dataset, evalDataset, numEpochs, lr, inputSize, hiddenSize, outputSize, model);
    }
}
