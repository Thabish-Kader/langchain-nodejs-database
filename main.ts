import { PineconeClient } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import * as dotenv from "dotenv";
import { createPineconeIndex } from "./createPineconeIndex";
import { updatePinecone } from "./updatePinecone";
import { queryPineconeVectorStoreAndQueryLLM } from "./queryPineconeAndQueryGpt";
import { Document } from "langchain/document";
dotenv.config();

// load documents from a ./documents
const loader = new DirectoryLoader("./documents", {
	".txt": (path) => new TextLoader(path),
	".pdf": (path) => new PDFLoader(path),
});
const docs: any = loader.load();

const question = "What is noahs ark ?";
const indexName = "langchain-index";
const vectorDimension = 1536;

// Initialize Pinecone
const client = new PineconeClient();

(async () => {
	await client.init({
		apiKey: process.env.PINECONE_API_KEY!,
		environment: process.env.PINECONE_ENVIRONMENT!,
	});
})();

(async () => {
	// 11. Check if Pinecone index exists and create if necessary
	await createPineconeIndex(client, indexName, vectorDimension);
	// 12. Update Pinecone vector store with document embeddings
	// await updatePinecone(client, indexName, docs);
	// 13. Query Pinecone vector store and GPT model for an answer
	await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
})();
