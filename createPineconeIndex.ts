import { PineconeClient } from "@pinecone-database/pinecone";

export const createPineconeIndex = async (
	client: PineconeClient,
	indexName: string,
	vectorDimension: number
) => {
	console.log(`--------> Checking ${indexName}`);
	const existingIndex = await client.listIndexes();

	if (!existingIndex.includes(indexName)) {
		console.log(`Createing ${indexName} --------->`);
		const createClient = await client.createIndex({
			createRequest: {
				name: indexName,
				dimension: vectorDimension,
				metric: "cosine",
			},
		});
		console.log(`-------> Created ${indexName}`);
		await new Promise((resolve) => setTimeout(resolve, 60000));
	} else {
		console.log(`-------> ${indexName} already exists`);
	}
};
