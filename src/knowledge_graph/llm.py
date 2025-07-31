"""LLM interaction utilities for knowledge graph generation."""
import requests
import json
import re

def call_llm(model, user_prompt, api_key, system_prompt=None, max_tokens=1000, temperature=0.2, base_url=None) -> str:
    """
    Call the language model API.
    
    Args:
        model: The model name to use
        user_prompt: The user prompt to send
        api_key: The API key for authentication
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        base_url: The base URL for the API endpoint
        
    Returns:
        The model's response as a string
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }
    
    messages = []
    
    if system_prompt:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })
    
    messages.append({
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': user_prompt
            }
        ]
    })
    
    payload = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    response = requests.post(
        base_url,
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API request failed: {response.text}")

def extract_json_from_text(text):
    """
    Extract JSON array from text that might contain additional content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        The parsed JSON if found, None otherwise
    """
    # First, check if the text is wrapped in code blocks with triple backticks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_match = re.search(code_block_pattern, text)
    if code_match:
        text = code_match.group(1).strip()
        print("Se encontró JSON en bloque de código, extrayendo contenido...")
    
    try:
        # Try direct parsing in case the response is already clean JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Look for opening and closing brackets of a JSON array
        start_idx = text.find('[')
        if start_idx == -1:
            print("No se encontró inicio de arreglo JSON en el texto")
            return None
            
        # Simple bracket counting to find matching closing bracket
        bracket_count = 0
        complete_json = False
        for i in range(start_idx, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    # Found the matching closing bracket
                    json_str = text[start_idx:i+1]
                    complete_json = True
                    break
        
        # Handle complete JSON array
        if complete_json:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("Se encontró estructura tipo JSON pero no se pudo analizar.")
                print("Intentando corregir problemas comunes de formato...")
                
                # Try to fix missing quotes around keys
                fixed_json = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', json_str)
                # Fix trailing commas
                fixed_json = re.sub(r',(\s*[\]}])', r'\1', fixed_json)
                
                try:
                    return json.loads(fixed_json)
                except:
                    print("No se pudieron corregir los problemas de formato JSON")
        else:
            # Handle incomplete JSON - try to complete it
            print("Se encontró arreglo JSON incompleto, intentando completarlo...")
            
            # Get all complete objects from the array
            objects = []
            obj_start = -1
            obj_end = -1
            brace_count = 0
            
            # First find all complete objects
            for i in range(start_idx + 1, len(text)):
                if text[i] == '{':
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_end = i
                        objects.append(text[obj_start:obj_end+1])
            
            if objects:
                # Reconstruct a valid JSON array with complete objects
                reconstructed_json = "[\n" + ",\n".join(objects) + "\n]"
                try:
                    return json.loads(reconstructed_json)
                except json.JSONDecodeError:
                    print("No se pudo analizar el arreglo JSON reconstruido.")
                    print("Intentando corregir problemas comunes de formato...")
                    
                    # Try to fix missing quotes around keys
                    fixed_json = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', reconstructed_json)
                    # Fix trailing commas
                    fixed_json = re.sub(r',(\s*[\]}])', r'\1', fixed_json)
                    
                    try:
                        return json.loads(fixed_json)
                    except:
                        print("No se pudieron corregir los problemas de formato JSON en el arreglo reconstruido")
            
        print("No se pudo extraer ningún arreglo JSON completo")
        return None 