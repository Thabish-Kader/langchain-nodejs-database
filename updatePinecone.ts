import { PineconeClient } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/dist/document";

export const updatePinecone = async (
	client: PineconeClient,
	indexName: string,
	docs: Document[]
) => {
	console.log(`retrieving Pinecone index...`);

	const index = client.Index(indexName);

	console.log(`Pinecone index retrieved: ${indexName}`);

	for (const doc of docs) {
		console.log(`Processing document : ${doc.metadata.source}`);
		const txtPath = doc.metadata.source;
		const text = doc.pageContent;

		const textSplitter = new RecursiveCharacterTextSplitter({
			chunkSize: 1000,
		});
		console.log(`Splitting text into chunks...`);

		const chunks = await textSplitter.createDocuments([text]);
		console.log(`Chunks created: ${chunks.length}`);
		console.log(
			`Calling OpenAis embedding endpoint document with ${chunks.length} text chunks...`
		);

		const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
			chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
		);

		// 9. Create and upsert vectors in batches of 100
		const batchSize = 100;
		let batch: any = [];
		for (let idx = 0; idx < chunks.length; idx++) {
			const chunk = chunks[idx];
			const vector = {
				id: `${txtPath}_${idx}`,
				values: embeddingsArrays[idx],
				metadata: {
					...chunk.metadata,
					loc: JSON.stringify(chunk.metadata.loc),
					pageContent: chunk.pageContent,
					txtPath: txtPath,
				},
			};
			batch.push(vector);
			// When batch is full or it's the last item, upsert the vectors
			if (batch.length === batchSize || idx === chunks.length - 1) {
				await index.upsert({
					upsertRequest: {
						vectors: batch,
					},
				});
				// Empty the batch
				batch = [];
			}
		}
		console.log(`Pinecone index updated: ${indexName.length} `);
	}
};
