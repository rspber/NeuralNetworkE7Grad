package nn.layer;

public class Main {

	public static void main(final String[] args) {

		// some data from air

		final int[] layerSizes = {1,1,1};
		
		final double learnRate = 0.05;

		final DataPoint[] trainingData = new DataPoint[] {
			new DataPoint(new double[] {0.5}, new double[] {0.5}),
			new DataPoint(new double[] {0.1}, new double[] {0.1}),
			new DataPoint(new double[] {0.3}, new double[] {0.3})
		};

		final NeuralNetwork nn = new NeuralNetwork(layerSizes);

		nn.learn(trainingData, learnRate);

		final double multipleCost = nn.calcMultipleCost(trainingData);

		System.out.println(multipleCost);
	}

}
