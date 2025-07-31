"""
Knowledge Graph Generator and Visualizer main module.
"""
import argparse
import json
import os
import sys

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge_graph.config import load_config
from src.knowledge_graph.llm import call_llm, extract_json_from_text
from src.knowledge_graph.visualization import visualize_knowledge_graph, sample_data_visualization
from src.knowledge_graph.text_utils import chunk_text
from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, limit_predicate_length
from src.knowledge_graph.text_processing import get_text_processor
from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT, MAIN_USER_PROMPT

def process_with_llm(config, input_text, debug=False):
    """
    Process input text with LLM to extract triples.
    
    Args:
        config: Configuration dictionary
        input_text: Text to analyze
        debug: If True, print detailed debug information
        
    Returns:
        List of extracted triples or None if processing failed
    """
    # Use prompts from the prompts module
    system_prompt = MAIN_SYSTEM_PROMPT
    user_prompt = MAIN_USER_PROMPT
    user_prompt += f"```\n{input_text}```\n" 

    # LLM configuration
    model = config["llm"]["model"]
    api_key = config["llm"]["api_key"]
    max_tokens = config["llm"]["max_tokens"]
    temperature = config["llm"]["temperature"]
    base_url = config["llm"]["base_url"]
    
    # Process with LLM
    metadata = {}
    response = call_llm(model, user_prompt, api_key, system_prompt, max_tokens, temperature, base_url)
    
    # Print raw response only if debug mode is on
    if debug:
        print("Respuesta cruda del LLM:")
        print(response)
        print("\n---\n")
    
    # Extract JSON from the response
    result = extract_json_from_text(response)
    
    if result:
        # Validate and filter triples to ensure they have all required fields
        valid_triples = []
        invalid_count = 0
        
        for item in result:
            if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                # Add metadata to valid items
                valid_triples.append(dict(item, **metadata))
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"Advertencia: Se filtraron {invalid_count} tripletes inválidos que carecían de campos requeridos")
        
        if not valid_triples:
            print("Error: No se encontraron tripletes válidos en la respuesta del LLM")
            return None
        
        # Apply predicate length limit to all valid triples
        for triple in valid_triples:
            triple["predicate"] = limit_predicate_length(triple["predicate"])
        
        # Filter triples using intelligent NLP-based approach
        text_processor = get_text_processor()
        filtered_triples = text_processor.filter_triples_intelligently(
            valid_triples, 
            source_texts=[input_text]
        )
        
        if len(filtered_triples) < len(valid_triples):
            removed_count = len(valid_triples) - len(filtered_triples)
            print(f"Se removieron {removed_count} tripletes mediante filtrado inteligente")
        
        # Print extracted JSON only if debug mode is on
        if debug:
            print("JSON extraído:")
            print(json.dumps(filtered_triples, indent=2))  # Pretty print the JSON
        
        return filtered_triples
    else:
        # Always print error messages even if debug is off
        print("\n\nERROR ### No se pudo extraer JSON válido de la respuesta: ", response, "\n\n")
        return None

def process_text_in_chunks(config, full_text, debug=False):
    """
    Process a large text by breaking it into chunks with overlap,
    and then processing each chunk separately.
    
    Args:
        config: Configuration dictionary
        full_text: The complete text to process
        debug: If True, print detailed debug information
    
    Returns:
        List of all extracted triples from all chunks
    """
    # Get chunking parameters from config
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)
    
    # Split text into chunks
    text_chunks = chunk_text(full_text, chunk_size, overlap)
    
    print("=" * 50)
    print("FASE 1: EXTRACCIÓN INICIAL DE TRIPLETES")
    print("=" * 50)
    print(f"Procesando texto en {len(text_chunks)} fragmentos (tamaño: {chunk_size} palabras, superposición: {overlap} palabras)")
    
    # Initialize text processor with full text for better TF-IDF analysis
    text_processor = get_text_processor()
    if text_chunks:
        print("Analizando importancia de términos en todo el corpus...")
        text_processor.analyze_text_importance(text_chunks, min_tfidf=0.1)
    
    # Process each chunk
    all_results = []
    for i, chunk in enumerate(text_chunks):
        print(f"Procesando fragmento {i+1}/{len(text_chunks)} ({len(chunk.split())} palabras)")
        
        # Process the chunk with LLM
        chunk_results = process_with_llm(config, chunk, debug)
        
        if chunk_results:
            # Add chunk information to each triple
            for item in chunk_results:
                item["chunk"] = i + 1
            
            # Add to overall results
            all_results.extend(chunk_results)
        else:
            print(f"Advertencia: Falló la extracción de tripletes del fragmento {i+1}")
    
    print(f"\nSe extrajo un total de {len(all_results)} tripletes de todos los fragmentos")
    
    # Apply entity standardization if enabled
    if config.get("standardization", {}).get("enabled", False):
        print("\n" + "="*50)
        print("FASE 2: ESTANDARIZACIÓN DE ENTIDADES")
        print("="*50)
        print(f"Comenzando con {len(all_results)} tripletes y {len(get_unique_entities(all_results))} entidades únicas")
        
        all_results = standardize_entities(all_results, config)
        
        print(f"Después de la estandarización: {len(all_results)} tripletes y {len(get_unique_entities(all_results))} entidades únicas")
    
    # Apply relationship inference if enabled
    if config.get("inference", {}).get("enabled", False):
        print("\n" + "="*50)
        print("FASE 3: INFERENCIA DE RELACIONES")
        print("="*50)
        print(f"Comenzando con {len(all_results)} tripletes")
        
        # Count existing relationships
        relationship_counts = {}
        for triple in all_results:
            relationship_counts[triple["predicate"]] = relationship_counts.get(triple["predicate"], 0) + 1
        
        print("Top 5 tipos de relaciones antes de la inferencia:")
        for pred, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {pred}: {count} ocurrencias")
        
        all_results = infer_relationships(all_results, config)
        
        # Count relationships after inference
        relationship_counts_after = {}
        for triple in all_results:
            relationship_counts_after[triple["predicate"]] = relationship_counts_after.get(triple["predicate"], 0) + 1
        
        print("\nTop 5 tipos de relaciones después de la inferencia:")
        for pred, count in sorted(relationship_counts_after.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {pred}: {count} ocurrencias")
        
        # Count inferred relationships
        inferred_count = sum(1 for triple in all_results if triple.get("inferred", False))
        print(f"\nSe agregaron {inferred_count} relaciones inferidas")
        print(f"Grafo de conocimiento final: {len(all_results)} tripletes")
    
    return all_results

def get_unique_entities(triples):
    """
    Get the set of unique entities from the triples.
    
    Args:
        triples: List of triple dictionaries
        
    Returns:
        Set of unique entity names
    """
    entities = set()
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        if "subject" in triple:
            entities.add(triple["subject"])
        if "object" in triple:
            entities.add(triple["object"])
    return entities

def main():
    """Main entry point for the knowledge graph generator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generador y Visualizador de Grafos de Conocimiento')
    parser.add_argument('--test', action='store_true', help='Generar una visualización de prueba con datos de muestra')
    parser.add_argument('--config', type=str, default='config.toml', help='Ruta al archivo de configuración')
    parser.add_argument('--output', type=str, default='knowledge_graph.html', help='Ruta del archivo HTML de salida')
    parser.add_argument('--input', type=str, required=False, help='Ruta al archivo de texto de entrada (requerido a menos que se use --test)')
    parser.add_argument('--debug', action='store_true', help='Habilitar salida de depuración (respuestas crudas del LLM y JSON extraído)')
    parser.add_argument('--no-standardize', action='store_true', help='Deshabilitar estandarización de entidades')
    parser.add_argument('--no-inference', action='store_true', help='Deshabilitar inferencia de relaciones')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print(f"Falló al cargar la configuración de {args.config}. Saliendo.")
        return
    
    # If test flag is provided, generate a sample visualization
    if args.test:
        print("Generando visualización de datos de muestra...")
        sample_data_visualization(args.output, config=config)
        print(f"\nVisualización de muestra guardada en {args.output}")
        print(f"Para ver la visualización, abre el siguiente archivo en tu navegador:")
        print(f"file://{os.path.abspath(args.output)}")
        return
    
    # For normal processing, input file is required
    if not args.input:
        print("Error: --input es requerido a menos que se use --test")
        parser.print_help()
        return
    
    # Override configuration settings with command line arguments
    if args.no_standardize:
        config.setdefault("standardization", {})["enabled"] = False
    if args.no_inference:
        config.setdefault("inference", {})["enabled"] = False
    
    # Load input text from file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
        print(f"Usando texto de entrada del archivo: {args.input}")
    except Exception as e:
        print(f"Error leyendo el archivo de entrada {args.input}: {e}")
        return
    
    # Process text in chunks
    result = process_text_in_chunks(config, input_text, args.debug)
    
    if result:
        # Save the raw data as JSON for potential reuse
        json_output = args.output.replace('.html', '.json')
        try:
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Datos crudos del grafo de conocimiento guardados en {json_output}")
        except Exception as e:
            print(f"Advertencia: No se pudieron guardar los datos crudos en {json_output}: {e}")
        
        # Visualize the knowledge graph
        stats = visualize_knowledge_graph(result, args.output, config=config)
        print("\nEstadísticas del Grafo de Conocimiento:")
        print(f"Nodos: {stats['nodes']}")
        print(f"Aristas: {stats['edges']}")
        print(f"Comunidades: {stats['communities']}")
        
        # Provide command to open the visualization in a browser
        print("\nPara ver la visualización, abre el siguiente archivo en tu navegador:")
        print(f"file://{os.path.abspath(args.output)}")
    else:
        print("La generación del grafo de conocimiento falló debido a errores en el procesamiento del LLM.")

if __name__ == "__main__":
    main() 