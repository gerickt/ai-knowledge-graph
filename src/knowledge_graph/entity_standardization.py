"""Entity standardization and relationship inference for knowledge graphs."""
import re
from collections import defaultdict
from src.knowledge_graph.llm import call_llm

def get_spanish_stopwords():
    """
    Return a comprehensive set of Spanish stopwords.
    
    Returns:
        set: Set of Spanish stopwords for filtering
    """
    return {
        # Articles
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        # Prepositions
        "a", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin", "sobre", "tras",
        # Pronouns
        "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "me", "te", "se", "le", "les", "lo", "los", "la", "las", "nos", "os", "su", "sus", "mi", "mis", "tu", "tus",
        # Conjunctions
        "y", "o", "pero", "sino", "aunque", "porque", "que", "si", "como", "cuando", "donde", "mientras", "tanto", "tan",
        # Verbs (common auxiliary and copular verbs)
        "ser", "estar", "haber", "tener", "hacer", "poder", "deber", "ir", "venir", "ver", "dar", "saber", "querer", "decir", "es", "son", "fue", "fueron", "era", "eran", "será", "serán", "sea", "sean", "está", "están", "estaba", "estaban", "estará", "estarán", "esté", "estén", "ha", "han", "había", "habían", "habrá", "habrán", "haya", "hayan", "he", "has", "hay",
        # Adverbs
        "no", "ni", "sí", "también", "tampoco", "además", "muy", "más", "menos", "mucho", "poco", "bastante", "demasiado", "bien", "mal", "mejor", "peor", "aquí", "ahí", "allí", "cerca", "lejos", "arriba", "abajo", "delante", "detrás", "dentro", "fuera", "antes", "después", "ahora", "entonces", "luego", "siempre", "nunca", "jamás", "ya", "aún", "todavía", "ayer", "hoy", "mañana",
        # Quantifiers and determiners
        "todo", "toda", "todos", "todas", "algún", "alguna", "algunos", "algunas", "ningún", "ninguna", "ningunos", "ningunas", "otro", "otra", "otros", "otras", "mismo", "misma", "mismos", "mismas", "cada", "cualquier", "cualquiera", "varios", "varias", "cierto", "cierta", "ciertos", "ciertas",
        # Question words
        "qué", "quién", "quiénes", "cuál", "cuáles", "cuándo", "cómo", "dónde", "por qué", "para qué",
        # Common words that should be filtered
        "al", "como", "entre", "hacia", "hasta", "mediante", "durante", "dentro", "fuera", "cerca", "lejos", "donde", "cuando", "porque", "aunque", "mientras", "tanto", "tan"
    }
from src.knowledge_graph.prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT, 
    get_entity_resolution_user_prompt,
    RELATIONSHIP_INFERENCE_SYSTEM_PROMPT,
    get_relationship_inference_user_prompt,
    WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT,
    get_within_community_inference_user_prompt
)

def limit_predicate_length(predicate, max_words=3):
    """
    Enforce a maximum word limit on predicates.
    
    Args:
        predicate: The original predicate string
        max_words: Maximum number of words allowed (default: 3)
        
    Returns:
        Shortened predicate with no more than max_words
    """
    words = predicate.split()
    if len(words) <= max_words:
        return predicate
    
    # If too long, use only the first max_words words
    shortened = ' '.join(words[:max_words])
    
    # Remove trailing prepositions or articles if they're the last word
    stop_words = get_spanish_stopwords()
    last_word = shortened.split()[-1].lower()
    if last_word in stop_words and len(words) > 1:
        shortened = ' '.join(shortened.split()[:-1])
    
    return shortened

def standardize_entities(triples, config):
    """
    Standardize entity names across all triples.
    
    Args:
        triples: List of dictionaries with 'subject', 'predicate', and 'object' keys
        config: Configuration dictionary
        
    Returns:
        List of triples with standardized entity names
    """
    if not triples:
        return triples
    
    print("Estandarizando nombres de entidades en todos los tripletes...")
    
    # Validate input triples to ensure they have the required fields
    valid_triples = []
    invalid_count = 0
    
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            valid_triples.append(triple)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Advertencia: Se filtraron {invalid_count} tripletes inválidos que carecían de campos requeridos")
    
    if not valid_triples:
        print("Error: No se encontraron tripletes válidos para la estandarización de entidades")
        return []
    
    # 1. Extract all unique entities
    all_entities = set()
    for triple in valid_triples:
        all_entities.add(triple["subject"].lower())
        all_entities.add(triple["object"].lower())
    
    # 2. Group similar entities - first by exact match after lowercasing and removing stopwords
    standardized_entities = {}
    entity_groups = defaultdict(list)
    
    # Helper function to normalize text for comparison
    def normalize_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove common stopwords that might appear in entity names
        stopwords = get_spanish_stopwords()
        words = [word for word in re.findall(r'\b\w+\b', text) if word not in stopwords]
        return " ".join(words)
    
    # Process entities in order of complexity (longer entities first)
    sorted_entities = sorted(all_entities, key=lambda x: (-len(x), x))
    
    # First pass: Standard normalization
    for entity in sorted_entities:
        normalized = normalize_text(entity)
        if normalized:  # Skip empty strings
            entity_groups[normalized].append(entity)
    
    # 3. For each group, choose the most representative name
    for group_key, variants in entity_groups.items():
        if len(variants) == 1:
            # Only one variant, use it directly
            standardized_entities[variants[0]] = variants[0]
        else:
            # Multiple variants, choose the most common or the shortest one as standard
            # Sort by frequency in triples, then by length (shorter is better)
            variant_counts = defaultdict(int)
            for triple in valid_triples:
                for variant in variants:
                    if triple["subject"].lower() == variant:
                        variant_counts[variant] += 1
                    if triple["object"].lower() == variant:
                        variant_counts[variant] += 1
            
            # Choose the most common variant as the standard form
            standard_form = sorted(variants, key=lambda x: (-variant_counts[x], len(x)))[0]
            for variant in variants:
                standardized_entities[variant] = standard_form
    
    # 4. Second pass: check for root word relationships
    # This handles cases like "capitalism" and "capitalist decay"
    additional_standardizations = {}
    
    # Get all standardized entity names (after first pass)
    standard_forms = set(standardized_entities.values())
    sorted_standards = sorted(standard_forms, key=len)
    
    for i, entity1 in enumerate(sorted_standards):
        e1_words = set(entity1.split())
        
        for entity2 in sorted_standards[i+1:]:
            if entity1 == entity2:
                continue
                
            # Check if one entity is a subset of the other
            e2_words = set(entity2.split())
            
            # If one entity contains all words from the other
            if e1_words.issubset(e2_words) and len(e1_words) > 0:
                # The shorter one is likely the more general concept
                additional_standardizations[entity2] = entity1
            elif e2_words.issubset(e1_words) and len(e2_words) > 0:
                additional_standardizations[entity1] = entity2
            else:
                # Check for stemming/root similarities
                stems1 = {word[:4] for word in e1_words if len(word) > 4}
                stems2 = {word[:4] for word in e2_words if len(word) > 4}
                
                shared_stems = stems1.intersection(stems2)
                
                if shared_stems and (len(shared_stems) / max(len(stems1), len(stems2))) > 0.5:
                    # Use the shorter entity as the standard
                    if len(entity1) <= len(entity2):
                        additional_standardizations[entity2] = entity1
                    else:
                        additional_standardizations[entity1] = entity2
    
    # Apply additional standardizations
    for entity, standard in additional_standardizations.items():
        standardized_entities[entity] = standard
    
    # 5. Apply standardization to all triples
    standardized_triples = []
    for triple in valid_triples:
        subj_lower = triple["subject"].lower()
        obj_lower = triple["object"].lower()
        
        standardized_triple = {
            "subject": standardized_entities.get(subj_lower, triple["subject"]),
            "predicate": limit_predicate_length(triple["predicate"]),
            "object": standardized_entities.get(obj_lower, triple["object"]),
            "chunk": triple.get("chunk", 0)
        }
        standardized_triples.append(standardized_triple)
    
    # 6. Optional: Use LLM to help with entity resolution for ambiguous cases
    if config.get("standardization", {}).get("use_llm_for_entities", False):
        standardized_triples = _resolve_entities_with_llm(standardized_triples, config)
    
    # 7. Filter out self-referencing triples
    filtered_triples = [triple for triple in standardized_triples if triple["subject"] != triple["object"]]
    if len(filtered_triples) < len(standardized_triples):
        print(f"Se removieron {len(standardized_triples) - len(filtered_triples)} tripletes auto-referenciados")
    
    print(f"Se estandarizaron {len(all_entities)} entidades en {len(set(standardized_entities.values()))} formas estándar")
    return filtered_triples

def infer_relationships(triples, config):
    """
    Infer additional relationships between entities to reduce isolated communities.
    
    Args:
        triples: List of dictionaries with standardized entity names
        config: Configuration dictionary
        
    Returns:
        List of triples with additional inferred relationships
    """
    if not triples or len(triples) < 2:
        return triples
    
    print("Infiriendo relaciones adicionales entre entidades...")
    
    # Validate input triples to ensure they have the required fields
    valid_triples = []
    invalid_count = 0
    
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            valid_triples.append(triple)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Advertencia: Se filtraron {invalid_count} tripletes inválidos que carecían de campos requeridos")
    
    if not valid_triples:
        print("Error: No se encontraron tripletes válidos para la inferencia de relaciones")
        return []
    
    # Create a graph representation for easier traversal
    graph = defaultdict(set)
    all_entities = set()
    for triple in valid_triples:
        subj = triple["subject"]
        obj = triple["object"]
        graph[subj].add(obj)
        all_entities.add(subj)
        all_entities.add(obj)
    
    # Find disconnected communities
    communities = _identify_communities(graph)
    print(f"Se identificaron {len(communities)} comunidades desconectadas en el grafo")
    
    new_triples = []
    
    # Use LLM to infer relationships between isolated communities if configured
    if config.get("inference", {}).get("use_llm_for_inference", True):
        # Infer relationships between different communities
        community_triples = _infer_relationships_with_llm(valid_triples, communities, config)
        if community_triples:
            new_triples.extend(community_triples)
            
        # Infer relationships within the same communities for semantically related entities
        within_community_triples = _infer_within_community_relationships(valid_triples, communities, config)
        if within_community_triples:
            new_triples.extend(within_community_triples)
    
    # Apply transitive inference rules
    transitive_triples = _apply_transitive_inference(valid_triples, graph)
    if transitive_triples:
        new_triples.extend(transitive_triples)
    
    # Infer relationships based on lexical similarity
    lexical_triples = _infer_relationships_by_lexical_similarity(all_entities, valid_triples)
    if lexical_triples:
        new_triples.extend(lexical_triples)
    
    # Add new triples to the original set
    if new_triples:
        valid_triples.extend(new_triples)
    
    # De-duplicate triples
    unique_triples = _deduplicate_triples(valid_triples)
    
    # Final pass: ensure all predicates follow the 3-word limit
    for triple in unique_triples:
        triple["predicate"] = limit_predicate_length(triple["predicate"])
    
    # Filter out self-referencing triples
    filtered_triples = [triple for triple in unique_triples if triple["subject"] != triple["object"]]
    if len(filtered_triples) < len(unique_triples):
        print(f"Se removieron {len(unique_triples) - len(filtered_triples)} tripletes auto-referenciados")
    
    print(f"Se agregaron {len(filtered_triples) - len(triples)} relaciones inferidas")
    return filtered_triples

def _identify_communities(graph):
    """
    Identify disconnected communities in the graph.
    
    Args:
        graph: Dictionary representing the graph structure
        
    Returns:
        List of sets, where each set contains nodes in a community
    """
    # Get all nodes
    all_nodes = set(graph.keys()).union(*[graph[node] for node in graph])
    
    # Track visited nodes
    visited = set()
    communities = []
    
    # Depth-first search to find connected components
    def dfs(node, community):
        visited.add(node)
        community.add(node)
        
        # Visit outgoing edges
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, community)
        
        # Visit incoming edges (we need to check all nodes)
        for source, targets in graph.items():
            if node in targets and source not in visited:
                dfs(source, community)
    
    # Find all communities
    for node in all_nodes:
        if node not in visited:
            community = set()
            dfs(node, community)
            communities.append(community)
    
    return communities

def _apply_transitive_inference(triples, graph):
    """
    Apply transitive inference to find new relationships.
    
    Args:
        triples: List of triple dictionaries
        graph: Dictionary representing the graph structure
        
    Returns:
        List of new inferred triples
    """
    new_triples = []
    
    # Predicates by subject-object pairs
    predicates = {}
    for triple in triples:
        key = (triple["subject"], triple["object"])
        predicates[key] = triple["predicate"]
    
    # Find transitive relationships: A -> B -> C implies A -> C
    for subj in graph:
        for mid in graph[subj]:
            for obj in graph.get(mid, []):
                # Only consider paths where A->B->C and A!=C
                if subj != obj and (subj, obj) not in predicates:
                    # Create a new predicate combining the two relationships
                    pred1 = predicates.get((subj, mid), "relates to")
                    pred2 = predicates.get((mid, obj), "relates to")
                    
                    # Generate a new predicate based on the transitive relationship
                    new_pred = f"indirectly {pred1}" if pred1 == pred2 else f"{pred1} via {mid}"
                    
                    # Add the new transitive relationship
                    new_triples.append({
                        "subject": subj,
                        "predicate": limit_predicate_length(new_pred),
                        "object": obj,
                        "inferred": True  # Mark as inferred
                    })
    
    return new_triples

def _deduplicate_triples(triples):
    """
    Remove duplicate triples, keeping the original (non-inferred) ones.
    
    Args:
        triples: List of triple dictionaries
        
    Returns:
        List of unique triples
    """
    # Use tuple of (subject, predicate, object) as key
    unique_triples = {}
    
    for triple in triples:
        key = (triple["subject"], triple["predicate"], triple["object"])
        # Keep original triples (not inferred) when duplicates exist
        if key not in unique_triples or not triple.get("inferred", False):
            unique_triples[key] = triple
    
    return list(unique_triples.values())

def _resolve_entities_with_llm(triples, config):
    """
    Use LLM to help resolve entity references and standardize entity names.
    
    Args:
        triples: List of triples with potentially non-standardized entities
        config: Configuration dictionary
        
    Returns:
        List of triples with LLM-assisted entity standardization
    """
    # Extract all unique entities
    all_entities = set()
    for triple in triples:
        all_entities.add(triple["subject"])
        all_entities.add(triple["object"])
    
    # If there are too many entities, limit to the most frequent ones
    if len(all_entities) > 100:
        # Count entity occurrences
        entity_counts = defaultdict(int)
        for triple in triples:
            entity_counts[triple["subject"]] += 1
            entity_counts[triple["object"]] += 1
        
        # Keep only the top 100 most frequent entities
        all_entities = {entity for entity, count in 
                       sorted(entity_counts.items(), key=lambda x: -x[1])[:100]}
    
    # Prepare prompt for LLM
    entity_list = "\n".join(sorted(all_entities))
    system_prompt = ENTITY_RESOLUTION_SYSTEM_PROMPT
    user_prompt = get_entity_resolution_user_prompt(entity_list)
    
    try:
        # LLM configuration
        model = config["llm"]["model"]
        api_key = config["llm"]["api_key"]
        max_tokens = config["llm"]["max_tokens"]
        temperature = config["llm"]["temperature"]
        base_url = config["llm"]["base_url"]
        
        # Call LLM
        response = call_llm(model, user_prompt, api_key, system_prompt, max_tokens, temperature, base_url)
        
        # Extract JSON mapping
        import json
        from src.knowledge_graph.llm import extract_json_from_text
        
        entity_mapping = extract_json_from_text(response)
        
        if entity_mapping and isinstance(entity_mapping, dict):
            # Apply the mapping to standardize entities
            entity_to_standard = {}
            for standard, variants in entity_mapping.items():
                for variant in variants:
                    entity_to_standard[variant] = standard
                # Also map the standard form to itself
                entity_to_standard[standard] = standard
            
            # Apply standardization to triples
            for triple in triples:
                triple["subject"] = entity_to_standard.get(triple["subject"], triple["subject"])
                triple["object"] = entity_to_standard.get(triple["object"], triple["object"])
                
            print(f"Se aplicó estandarización de entidades basada en LLM para {len(entity_mapping)} grupos de entidades")
        else:
            print("No se pudo extraer mapeo válido de entidades de la respuesta del LLM")
    
    except Exception as e:
        print(f"Error en la resolución de entidades basada en LLM: {e}")
    
    return triples

def _infer_relationships_with_llm(triples, communities, config):
    """
    Use LLM to infer relationships between disconnected communities.
    
    Args:
        triples: List of existing triples
        communities: List of community sets
        config: Configuration dictionary
        
    Returns:
        List of new inferred triples
    """
    # Skip if there's only one community
    if len(communities) <= 1:
        print("Solo se encontró una comunidad, omitiendo la inferencia de relaciones basada en LLM")
        return []
    
    # Focus on the largest communities
    large_communities = sorted(communities, key=len, reverse=True)[:5]
    
    # For each pair of large communities, try to infer relationships
    new_triples = []
    
    for i, comm1 in enumerate(large_communities):
        for j, comm2 in enumerate(large_communities):
            if i >= j:
                continue  # Skip self-comparisons and duplicates
            
            # Select representative entities from each community
            rep1 = list(comm1)[:min(5, len(comm1))]
            rep2 = list(comm2)[:min(5, len(comm2))]
            
            # Prepare relevant existing triples for context
            context_triples = []
            for triple in triples:
                if triple["subject"] in rep1 or triple["subject"] in rep2 or \
                   triple["object"] in rep1 or triple["object"] in rep2:
                    context_triples.append(triple)
            
            # Limit context size
            if len(context_triples) > 20:
                context_triples = context_triples[:20]
            
            # Convert triples to text for prompt
            triples_text = "\n".join([
                f"{t['subject']} {t['predicate']} {t['object']}"
                for t in context_triples
            ])
            
            # Prepare entity lists
            entities1 = ", ".join(rep1)
            entities2 = ", ".join(rep2)
            
            # Create prompt for LLM
            system_prompt = RELATIONSHIP_INFERENCE_SYSTEM_PROMPT
            user_prompt = get_relationship_inference_user_prompt(entities1, entities2, triples_text)
            
            try:
                # LLM configuration
                model = config["llm"]["model"]
                api_key = config["llm"]["api_key"]
                max_tokens = config["llm"]["max_tokens"]
                temperature = config["llm"]["temperature"]
                base_url = config["llm"]["base_url"]
                
                # Call LLM
                response = call_llm(model, user_prompt, api_key, system_prompt, max_tokens, temperature, base_url)
                
                # Extract JSON results
                from src.knowledge_graph.llm import extract_json_from_text
                inferred_triples = extract_json_from_text(response)
                
                if inferred_triples and isinstance(inferred_triples, list):
                    # Mark as inferred and add to new triples
                    for triple in inferred_triples:
                        if "subject" in triple and "predicate" in triple and "object" in triple:
                            # Skip self-referencing triples
                            if triple["subject"] == triple["object"]:
                                continue
                            triple["inferred"] = True
                            triple["predicate"] = limit_predicate_length(triple["predicate"])
                            new_triples.append(triple)
                    
                    print(f"Se infirieron {len(new_triples)} nuevas relaciones entre comunidades")
                else:
                    print("No se pudieron extraer relaciones inferidas válidas de la respuesta del LLM")
            
            except Exception as e:
                print(f"Error en la inferencia de relaciones basada en LLM: {e}")
    
    return new_triples 

def _infer_within_community_relationships(triples, communities, config):
    """
    Use LLM to infer relationships between entities within the same community.
    Focus on entities that might be semantically related but not directly connected.
    
    Args:
        triples: List of existing triples
        communities: List of community sets
        config: Configuration dictionary
        
    Returns:
        List of new inferred triples
    """
    new_triples = []
    
    # Process larger communities
    for community in sorted(communities, key=len, reverse=True)[:3]:
        # Skip small communities
        if len(community) < 5:
            continue
            
        # Get all entities in this community
        community_entities = list(community)
        
        # Create an adjacency matrix to identify disconnected entity pairs
        connections = {(a, b): False for a in community_entities for b in community_entities if a != b}
        
        # Mark existing connections
        for triple in triples:
            if triple["subject"] in community_entities and triple["object"] in community_entities:
                connections[(triple["subject"], triple["object"])] = True
        
        # Find disconnected pairs that might be semantically related
        disconnected_pairs = []
        for (a, b), connected in connections.items():
            if not connected and a != b:  # Ensure a and b are different entities
                # Check for potential semantic relationship (e.g., shared words)
                a_words = set(a.lower().split())
                b_words = set(b.lower().split())
                shared_words = a_words.intersection(b_words)
                
                # If they share words or one is contained in the other, they might be related
                if shared_words or a.lower() in b.lower() or b.lower() in a.lower():
                    disconnected_pairs.append((a, b))
        
        # Limit to the most promising pairs
        disconnected_pairs = disconnected_pairs[:10]
        
        if not disconnected_pairs:
            continue
            
        # Get relevant context
        context_triples = []
        entities_of_interest = set()
        for a, b in disconnected_pairs:
            entities_of_interest.add(a)
            entities_of_interest.add(b)
            
        for triple in triples:
            if triple["subject"] in entities_of_interest or triple["object"] in entities_of_interest:
                context_triples.append(triple)
        
        # Limit context size
        if len(context_triples) > 20:
            context_triples = context_triples[:20]
            
        # Convert triples to text for prompt
        triples_text = "\n".join([
            f"{t['subject']} {t['predicate']} {t['object']}"
            for t in context_triples
        ])
        
        # Create pairs text
        pairs_text = "\n".join([f"{a} and {b}" for a, b in disconnected_pairs])
        
        # Create prompt for LLM
        system_prompt = WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT
        user_prompt = get_within_community_inference_user_prompt(pairs_text, triples_text)
        
        try:
            # LLM configuration
            model = config["llm"]["model"]
            api_key = config["llm"]["api_key"]
            max_tokens = config["llm"]["max_tokens"]
            temperature = config["llm"]["temperature"]
            base_url = config["llm"]["base_url"]
            
            # Call LLM
            response = call_llm(model, user_prompt, api_key, system_prompt, max_tokens, temperature, base_url)
            
            # Extract JSON results
            from src.knowledge_graph.llm import extract_json_from_text
            inferred_triples = extract_json_from_text(response)
            
            if inferred_triples and isinstance(inferred_triples, list):
                # Mark as inferred and add to new triples
                for triple in inferred_triples:
                    if "subject" in triple and "predicate" in triple and "object" in triple:
                        # Skip self-referencing triples
                        if triple["subject"] == triple["object"]:
                            continue
                        triple["inferred"] = True
                        triple["predicate"] = limit_predicate_length(triple["predicate"])
                        new_triples.append(triple)
                
                print(f"Se infirieron {len(inferred_triples)} nuevas relaciones dentro de las comunidades")
            else:
                print("Could not extract valid inferred relationships from LLM response")
        
        except Exception as e:
            print(f"Error en la inferencia de relaciones basada en LLM dentro de las comunidades: {e}")
    
    return new_triples

def _infer_relationships_by_lexical_similarity(entities, triples):
    """
    Infer relationships between entities based on lexical similarity.
    This can help connect entities like "capitalism" and "capitalist decay".
    
    Args:
        entities: Set of all entities
        triples: List of existing triples
        
    Returns:
        List of new inferred triples
    """
    new_triples = []
    processed_pairs = set()
    
    # Create a dictionary to track existing relationships
    existing_relationships = set()
    for triple in triples:
        existing_relationships.add((triple["subject"], triple["object"]))
    
    # Check for lexical similarity between entities
    entities_list = list(entities)
    for i, entity1 in enumerate(entities_list):
        for entity2 in entities_list[i+1:]:
            # Skip if already connected
            if (entity1, entity2) in existing_relationships or (entity2, entity1) in existing_relationships:
                continue
                
            # Skip if already processed this pair
            if (entity1, entity2) in processed_pairs or (entity2, entity1) in processed_pairs:
                continue
                
            # Skip if the entities are the same (prevent self-reference)
            if entity1 == entity2:
                continue
                
            processed_pairs.add((entity1, entity2))
            
            # Check for containment or shared roots
            e1_lower = entity1.lower()
            e2_lower = entity2.lower()
            
            # Simple word overlap check
            e1_words = set(e1_lower.split())
            e2_words = set(e2_lower.split())
            shared_words = e1_words.intersection(e2_words)
            
            if shared_words:
                # Create relationships based on shared words
                main_shared = max(shared_words, key=len)
                
                if len(main_shared) >= 4:  # Only consider significant shared words
                    if e1_lower.startswith(main_shared) and not e2_lower.startswith(main_shared):
                        new_triples.append({
                            "subject": entity2,
                            "predicate": "se relaciona con",
                            "object": entity1,
                            "inferred": True
                        })
                    elif e2_lower.startswith(main_shared) and not e1_lower.startswith(main_shared):
                        new_triples.append({
                            "subject": entity1,
                            "predicate": "se relaciona con",
                            "object": entity2,
                            "inferred": True
                        })
                    else:
                        new_triples.append({
                            "subject": entity1,
                            "predicate": "relacionado con",
                            "object": entity2,
                            "inferred": True
                        })
            
            # Check if one entity contains the other
            elif e1_lower in e2_lower:
                new_triples.append({
                    "subject": entity2,
                    "predicate": "es tipo de",
                    "object": entity1,
                    "inferred": True
                })
            elif e2_lower in e1_lower:
                new_triples.append({
                    "subject": entity1,
                    "predicate": "es tipo de",
                    "object": entity2,
                    "inferred": True
                })
    
    print(f"Se infirieron {len(new_triples)} relaciones basadas en similitud léxica")
    return new_triples 