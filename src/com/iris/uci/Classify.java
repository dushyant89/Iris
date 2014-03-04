package com.iris.uci;

import com.iris.uci.Layer;
import com.iris.uci.Link;
import com.iris.uci.Neuron;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

public class Classify 
{

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub

		System.out.println("Creating feature vectors for the iris data");
		FileInputStream fis = null;
		BufferedReader reader = null;
		double e=2.71828182846;
		try 
		{
			fis = new FileInputStream("C:/Dushyant/Artificial Intelligence/Iris/iris_data");
			reader = new BufferedReader(new InputStreamReader(fis));
			String line = reader.readLine();
			int rows=0;
			while(line!=null)
			{
				//reading the no. of rows for creating the truth table
				rows++;
				line=reader.readLine();
			}
			String [][]table_strings=new String [rows][5];
			fis.close();
			reader.close();

			fis = new FileInputStream("C:/Dushyant/Artificial Intelligence/Iris/iris_data");
			reader = new BufferedReader(new InputStreamReader(fis));
			line = reader.readLine();
			rows=0;
			while(line !=null)
			{
				String []values=line.split(",");
				for(int i=0;i<5;i++)
				{
					table_strings[rows][i]=values[i];
				}
				rows++;
				line=reader.readLine();
			}
			System.out.println("Truth table is prepared, changing the values to double");
			double [][]truth_table=new double[rows][5];
			for(int i=0;i<rows;i++)
			{
				for(int j=0;j<4;j++)
					truth_table[i][j]=Double.parseDouble(table_strings[i][j]);
				
				if(table_strings[i][4].equalsIgnoreCase("Iris-setosa"))
					truth_table[i][4]=1;
				
				else if(table_strings[i][4].equalsIgnoreCase("Iris-versicolor"))
					truth_table[i][4]=2;
				
				else
					truth_table[i][4]=3;
			}
			fis.close();
			reader.close();
			System.out.println("Truth table is prepared");

			System.out.println("Creating the neural network architecture now:Enter configuration details");

			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			int hidden_layers;
			System.out.println("Enter the no. of hidden layers you want");
			hidden_layers=Integer.parseInt(br.readLine());

			int no_neurons;
			System.out.println("Enter the no of neurons per hidden layer");
			no_neurons=Integer.parseInt(br.readLine());

			double learning_rate;
			System.out.println("Enter the starting learning rate");
			learning_rate=Double.parseDouble(br.readLine());

			double momentum;
			System.out.println("Enter the starting momentum");
			momentum=Double.parseDouble(br.readLine());
			
			int cycles=40000;
			System.out.println("Enter the no. of cycles you want the network to train for");
			cycles=Integer.parseInt(br.readLine());
			
			//input layer neurons+ last layer+ hidden layer
			int total_links=4*no_neurons + 3*no_neurons +(hidden_layers-1)*no_neurons*no_neurons;
			int link_tracker=0;
			//generating the link weights randomly for the first iteration 
			Double[] links=new Double[total_links];

			for(int i=0;i<total_links;i++)
			{
				links[i]=Math.random();
			}
			Layer []layers=new Layer[hidden_layers+1];
			//assigning the link weights to the input layer

			layers[0]=new Layer();
			layers[0].neurons=new Neuron[no_neurons];
			for(int i=0;i<no_neurons;i++)
			{
				layers[0].neurons[i]=new Neuron();
				layers[0].neurons[i].to=new Link[4];
				for(int j=0;j<4;j++)
				{
					layers[0].neurons[i].to[j]=new Link();
					layers[0].neurons[i].to[j].weight=links[link_tracker++];
				}
			}  

			//assigning weights to other hidden layer neurons
			for(int i=1;i<hidden_layers;i++)
			{
				layers[i]=new Layer();
				layers[i].neurons=new Neuron[no_neurons];
				for(int j=0;j<no_neurons;j++)
				{
					layers[i].neurons[j]=new Neuron();
					layers[i].neurons[j].to=new Link[no_neurons];
					for(int k=0;k<no_neurons;k++)
					{
						layers[i].neurons[j].to[k]=new Link();
						layers[i].neurons[j].to[k].weight=links[link_tracker++];
					}
				}
			}

			//assigning weights to last year
			layers[hidden_layers]=new Layer();
			layers[hidden_layers].neurons=new Neuron[3];
			for(int i=0;i<3;i++)
			{
				layers[hidden_layers].neurons[i]= new Neuron();
				layers[hidden_layers].neurons[i].to=new Link[no_neurons];
				for(int j=0;j<no_neurons;j++)
				{
					layers[hidden_layers].neurons[i].to[j]=new Link();
					layers[hidden_layers].neurons[i].to[j].weight=links[link_tracker++];
				}
			}

			//done with the initializing of weights

			int run_count=0;
			int check=0;
			outer:while(true)
			{	 
				run_count++;
				
				if(run_count >cycles) break;
				
				for(int row=0;row < rows;row++)
				{
					for(int i=0;i<=hidden_layers;i++)
					{
						if(i==0) // the first layer of the network connected to inputs
						{
							double net=0;
							for(int n=0;n<no_neurons;n++)
							{
								for(int j=0;j<4;j++)
								{
									net+=truth_table[row][j]*layers[i].neurons[n].to[j].weight;
								}
								double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
								layers[i].neurons[n].output=op;
							}
						}
						else if(i==hidden_layers) //the last layer of the network
						{
							double net=0;
							for(int n=0;n<3;n++)
							{
								for(int j=0;j<no_neurons;j++)
								{
									net+=layers[i-1].neurons[j].output*layers[i].neurons[n].to[j].weight;
								}
								double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
								layers[i].neurons[n].output=op;
							} 
						}
						else
						{
							double net=0;
							for(int n=0;n<no_neurons;n++)
							{
								for(int j=0;j<no_neurons;j++)
								{
									net+=layers[i-1].neurons[j].output*layers[i].neurons[n].to[j].weight;
								}
								double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
								layers[i].neurons[n].output=op;
							} 
						}
					}
					/*
					 * Here for learning what we are forcing is that the o/p neuron 0 should be on +ve positive tweet
					 * o/p neuron 1 for -ve tweets
					 * o/p neuron 2 for objective tweets
					 */
					int t;
					double o;
					double E=0.0;
					for(int i=0;i<3;i++)
					{
						if(i==0 && truth_table[row][4]==1.0) 
						{
							t=1;
						}
						else if(i==1 && truth_table[row][4]==2.0)
						{
							t=1;
						}
						else if(i==2 && truth_table[row][4]==3.0)
						{
							t=1;
						}
						else
						{
							t=0;
						}
						o=layers[hidden_layers].neurons[i].output;
						layers[hidden_layers].neurons[i].error=(t-o)*o*(1-o);
						E=E+Math.abs(t-o);
					}
					//calculating the total sum squared error
					E=Math.pow(E,2);
					E=E/2;
					E=E*100;
					E=Math.round(E);
					E=E/100;
					System.out.println(E+" "+run_count+" "+row+ " "+check);
					if(E <= 0.06)
					{
						check++;
						if(check==rows)
						{
							System.out.println("We are done with "+run_count+" run(s)");
							break outer;
						}
						else
						{
							continue;
						}
					}
					else
					{
						/**
						 * backpropagate for online learning
						 */
						check=0;
						for(int i=hidden_layers;i>=0;i--)
						{
							if(i==hidden_layers) //the last layer with three single neurons
							{
								for(int j=0;j<3;j++)
								{	
									for(int n=0;n<no_neurons;n++)
									{
										layers[i].neurons[j].to[n].weight+=learning_rate*layers[i].neurons[j].error*layers[i-1].neurons[n].output + momentum*layers[i].neurons[j].to[n].delta_weight;
										layers[i].neurons[j].to[n].delta_weight=learning_rate*layers[i].neurons[j].error*layers[i-1].neurons[n].output;
									}
								}	
							}
							else if(i==0)  //the first layer
							{
								for(int n=0;n<no_neurons;n++) //neurons for the layer whose incoming weights are to be corrected
								{
									o=layers[i].neurons[n].output;
									for(int j=0;j<layers[i].neurons[n].to.length;j++) //links coming towards the neuron whose weights are to be corrected
									{
										for(int k=0;k<layers[i+1].neurons.length;k++)
										{
											//calculating the error from the links going outward and the next layers error
											layers[i].neurons[n].error+=layers[i+1].neurons[k].to[n].weight*layers[i+1].neurons[k].error;
										}
										layers[i].neurons[n].error*=(1-o)*o;
										layers[i].neurons[n].to[j].weight+=learning_rate*layers[i].neurons[n].error*truth_table[row][j] + momentum*layers[i].neurons[n].to[j].delta_weight; 
										layers[i].neurons[n].to[j].delta_weight=learning_rate*layers[i].neurons[n].error*truth_table[row][j];
									}

								}
							}
							else  //the case otherwise
							{
								//same as above only the output of the input layer has to be changed and one iteration will be less
								for(int n=0;n<no_neurons;n++) //neurons for the layer whose incoming weights are to be corrected
								{
									o=layers[i].neurons[n].output;
									for(int j=0;j<layers[i].neurons[n].to.length;j++) //links coming towards the neuron whose weights are to be corrected
									{
										for(int k=0;k<layers[i+1].neurons.length;k++)
										{
											//calculating the error from the links going outward and the next layers error
											layers[i].neurons[n].error+=layers[i+1].neurons[k].to[n].weight*layers[i+1].neurons[k].error;
										}
										layers[i].neurons[n].error*=(1-o)*o;
										layers[i].neurons[n].to[j].weight+=learning_rate*layers[i].neurons[n].error*layers[i-1].neurons[j].output +  momentum*layers[i].neurons[n].to[j].delta_weight;
										layers[i].neurons[n].to[j].delta_weight=learning_rate*layers[i].neurons[n].error*layers[i-1].neurons[j].output;  
									}

								}
							}
						}
					}
				}
			} 

			System.out.println("Now that the network has converged , lets make efforts to generalize it");
			
			fis = new FileInputStream("C:/Dushyant/Artificial Intelligence/Iris/iris_data_test");
			reader = new BufferedReader(new InputStreamReader(fis));
			line = reader.readLine();
			rows=0;
			while(line!=null)
			{
				//reading the no. of rows for creating the truth table
				rows++;
				line=reader.readLine();
			}
			System.out.println(rows);
			fis.close();
			reader.close();
			
			String [][]table_test_strings=new String [rows][5];
			
			fis = new FileInputStream("C:/Dushyant/Artificial Intelligence/Iris/iris_data_test");
			reader = new BufferedReader(new InputStreamReader(fis));
			line = reader.readLine();
			rows=0;
			while(line !=null)
			{
				String []values=line.split(",");
				for(int i=0;i<5;i++)
				{
					table_test_strings[rows][i]=values[i];
				}
				rows++;
				line=reader.readLine();
			}
			System.out.println("Test Truth table is prepared, changing the values to double");
			double [][]truth_table_test=new double[rows][5];
			
			for(int i=0;i<rows;i++)
			{
				for(int j=0;j<4;j++)
					truth_table_test[i][j]=Double.parseDouble(table_test_strings[i][j]);
				
				if(table_test_strings[i][4].equalsIgnoreCase("Iris-setosa"))
					truth_table_test[i][4]=1;
				
				else if(table_test_strings[i][4].equalsIgnoreCase("Iris-versicolor"))
					truth_table_test[i][4]=2;
				
				else
					truth_table_test[i][4]=3;
			}
			fis.close();
			reader.close();
			System.out.println("Test Truth table is prepared");
			
			int error_count=0;
			
			for(int row=0;row < rows;row++)
			{
				for(int i=0;i<=hidden_layers;i++)
				{
					if(i==0) // the first layer of the network connected to inputs
					{
						double net=0;
						for(int n=0;n<no_neurons;n++)
						{
							for(int j=0;j<4;j++)
							{
								net+=truth_table_test[row][j]*layers[i].neurons[n].to[j].weight;
							}
							double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
							layers[i].neurons[n].output=op;
						}
					}
					else if(i==hidden_layers) //the last layer of the network
					{
						double net=0;
						for(int n=0;n<3;n++)
						{
							for(int j=0;j<no_neurons;j++)
							{
								net+=layers[i-1].neurons[j].output*layers[i].neurons[n].to[j].weight;
							}
							double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
							layers[i].neurons[n].output=op;
						} 
					}
					else
					{
						double net=0;
						for(int n=0;n<no_neurons;n++)
						{
							for(int j=0;j<no_neurons;j++)
							{
								net+=layers[i-1].neurons[j].output*layers[i].neurons[n].to[j].weight;
							}
							double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
							layers[i].neurons[n].output=op;
						} 
					}
				}
				/*
				 * Here for learning what we are forcing is that the o/p neuron 0 should be on +ve positive tweet
				 * o/p neuron 1 for -ve tweets
				 * o/p neuron 2 for objective tweets
				 */
				double o,max=0;
				for(int i=0;i<3;i++)
				{
					o=layers[hidden_layers].neurons[i].output;
					if(o >max)
					{
						max=o;
					}
				}
				for(int i=0;i<3;i++)
				{
					System.out.println("max:"+max+" "+i+" "+layers[hidden_layers].neurons[i].output+" "+truth_table_test[row][4]);
					
					if(i==0 && truth_table_test[row][4]==1.0) 
					{
						if(max==layers[hidden_layers].neurons[i].output)
						{
							break;
						}
						else
						{
							error_count++;
							break;
						}
					}
					else if(i==1 && truth_table_test[row][4]==2.0)
					{
						if(max==layers[hidden_layers].neurons[i].output)
						{
							break;
						}
						else
						{
							error_count++;
							break;
						}
					}
					else if(i==2 && truth_table_test[row][4]==3.0)
					{
						if(max==layers[hidden_layers].neurons[i].output)
						{
							break;
						}
						else
						{
							error_count++;
							break;
						}
					}
				}
				System.out.println();
			}	
			System.out.print("% accuracy is:");
			System.out.println((1.00-((double)error_count/(double)rows))*100);

		} catch (FileNotFoundException e1) 
		{
			// TODO Auto-generated catch block
			System.out.println("file not found, enter the path correctly");
		} catch (IOException e2) 
		{
			// TODO Auto-generated catch block
			e2.printStackTrace();
		} catch(Exception e3)
		{
			System.out.println("Some error occcured, try running again with a valid configuration");
		}

	}

}
