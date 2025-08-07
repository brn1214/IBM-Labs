import chromadb
from chromadb.utils import embedding_functions

# ef define la funcion para calcular los embeddings, usando SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = "all-MiniLM-L6-v2"
)

# Nueva instancia de chromaclient para interactuar con chroma db
client = chromadb.Client()
# Definir el nombre de la coleccion que se va a crear o retrieve
collection_name = "my_grocery_collection"

# Funcion principal que va a interactuar con ChromaDB
def main():
    try:
        # Crear la coleccion en chromadb, con su nombre, distancia y funcion de embedd. Aqui se usa 
        # distancia coseno
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A colection for storing grocery data"},
            configuration={
                "hnsw" : {"space": "cosine"},
                "embedding_function" : ef
            }
        )
        print(f"Collection created: {collection.name}")

        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        # Creando IDs unicos
        ids = [f"food_{index + 1}" for index,_ in enumerate(texts)]

        # Agregar los documentos con su correspondiente ID
        # ChromaDB va a generar automaticamente los embeddings basado en la configuración
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        # Retrieve items (documents) almacenados en la coleccion
        all_items = collection.get()
        # El siguiente codigo imprime todos los documentos, IDs, metadata
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        ###### SIMILARITY SEARCH #######
        # Funcion de busqueda similar
        def perform_similarity_search(collection, all_items):
            try:
                query_term = ["red", "fresh"]
                if isinstance(query_term, str):
                    query_term = [query_term]
                # Busqueda del query
                # Realizar la busqueda del documento mas similar
                results = collection.query(
                    query_texts=query_term,
                    n_results=3
                )
                print(f"Query results for '{query_term}'")
                print(results)

                ##### Mostrar los mejores resultados
                # Si no hay resultados
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print(f'No documents found similar to "{query_term}"')
                    return
                # Mostrar el top 3 resultados
                for q in range(len(query_term)):
                    print(f'Top 3 similar docs to "{query_term[q]}":')
                        
                    for i in range(min(3, len(results['ids'][q]))):
                        doc_id = results['ids'][q][i] # IDs
                        score = results['distances'][q][i]
                        text = results['documents'][q][i]

                        if not text:
                            print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                        else:
                            print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

            except Exception as error:
                print(f"Error in similarity search: {error}")

        perform_similarity_search(collection, all_items)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()