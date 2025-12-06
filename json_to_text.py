# json_to_text.py
import json
import os

class JSONTextExtractor:
    """Extraer texto de diferentes formatos JSON para enviar al LLM"""
    
    @staticmethod
    def extract_from_json(json_path):
        """
        Extraer texto de cualquier JSON (OCR o Speech)
        
        Args:
            json_path: Ruta al archivo JSON
        
        Returns:
            str: Texto para LLM (STRING, no guarda archivo)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Formato OCR (PDF/Imágenes)
            if isinstance(data, dict) and 'pages' in data:
                return JSONTextExtractor._extract_ocr_format(data)
            
            # Formato Speech-to-Text (Audio/Video)
            elif isinstance(data, list):
                return JSONTextExtractor._extract_speech_format(data)
            
            else:
                print(f"⚠️ Formato JSON no reconocido: {json_path}")
                return ""
                
        except Exception as e:
            print(f"❌ Error leyendo JSON {json_path}: {e}")
            return ""
    
    @staticmethod
    def extract_all_from_folder(folder_path):
        """
        Extraer texto de TODOS los JSONs en una carpeta
        
        Args:
            folder_path: Carpeta con archivos JSON
        
        Returns:
            str: Todo el texto concatenado (STRING, no guarda archivo)
        """
        all_texts = []
        
        print(f"🔍 Buscando JSONs en: {folder_path}")
        
        # Buscar todos los archivos JSON
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    print(f"  📄 Procesando: {file}")
                    
                    text = JSONTextExtractor.extract_from_json(json_path)
                    if text:
                        # Separador entre documentos
                        all_texts.append(f"\n{'='*80}\n")
                        all_texts.append(f"ARCHIVO: {file}\n")
                        all_texts.append(text)
        
        if not all_texts:
            return "❌ No se encontraron archivos JSON válidos"
        
        # Unir todos los textos (STRING)
        final_text = "".join(all_texts)
        
        print(f"✅ Texto extraído: {len(final_text)} caracteres")
        return final_text  # ← SOLO STRING, NO GUARDA ARCHIVO
    
    @staticmethod
    def _extract_ocr_format(data):
        """Extraer texto de formato OCR (PDF/Imágenes)"""
        text_parts = []
        
        # Metadata del documento
        filename = data.get('filename', 'Documento Desconocido')
        total_pages = data.get('total_pages', 0)
        
        text_parts.append(f"📄 DOCUMENTO: {filename}")
        if total_pages > 0:
            text_parts.append(f"📊 Páginas: {total_pages}")
        text_parts.append("=" * 50)
        
        # Procesar cada página
        for page in data.get('pages', []):
            page_num = page.get('page_number', '?')
            content_type = page.get('content_type', 'text')
            page_text = page.get('text', '').strip()
            
            if not page_text:
                continue
            
            # Encabezado de página
            if content_type == 'text':
                text_parts.append(f"\n[PÁGINA {page_num}]")
            else:
                text_parts.append(f"\n[PÁGINA {page_num} - {content_type.upper()}]")
            
            text_parts.append(page_text)
        
        # Resumen si existe
        if 'summary' in data:
            summary = data['summary']
            has_content = any(value > 0 for value in summary.values())
            
            if has_content:
                text_parts.append("\n" + "=" * 50)
                text_parts.append("📈 RESUMEN DEL DOCUMENTO:")
                
                for key, value in summary.items():
                    if value > 0:
                        # Traducir keys a español
                        key_es = {
                            'text_pages': 'Páginas de texto',
                            'id_card_pages': 'Páginas con ID',
                            'picture_pages': 'Páginas con imágenes',
                            'table_pages': 'Páginas con tablas'
                        }.get(key, key.replace('_', ' ').title())
                        text_parts.append(f"  • {key_es}: {value}")
        
        return "\n".join(text_parts)
    
    @staticmethod
    def _extract_speech_format(data, include_timestamps=True):
        """Extraer texto de formato Speech-to-Text (Audio/Video)"""
        if not data or not isinstance(data, list):
            return ""
        
        text_parts = []
        
        # Encabezado para audio/video
        text_parts.append("🎤 TRANSCRIPCIÓN DE AUDIO/VIDEO")
        text_parts.append("=" * 50)
        
        for segment in data:
            if not isinstance(segment, dict):
                continue
            
            # Tu formato específico (con 'content')
            if 'content' in segment:
                content = segment.get('content', '').strip()
                if content:
                    # Añadir timestamp si está disponible
                    if include_timestamps and 'start' in segment:
                        timestamp = JSONTextExtractor._format_timestamp(segment['start'])
                        text_parts.append(f"[{timestamp}] {content}")
                    else:
                        text_parts.append(content)
        
        # Estadísticas
        if data:
            text_parts.append(f"\n📊 Total segmentos: {len(data)}")
        
        return "\n".join(text_parts)
    
    @staticmethod
    def _format_timestamp(seconds):
        """Convertir segundos a formato MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

# Funciones de conveniencia (para mantener compatibilidad)
def extract_text_from_json(json_path):
    """Extraer texto de un solo JSON"""
    return JSONTextExtractor.extract_from_json(json_path)

def extract_text_from_folder(folder_path):
    """Extraer texto de todos los JSONs en una carpeta"""
    return JSONTextExtractor.extract_all_from_folder(folder_path)

# Para uso como script independiente
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        
        # Obtener string (no guardar archivo)
        llm_text = JSONTextExtractor.extract_all_from_folder(folder)
        
        print("\n" + "="*80)
        print("🎯 TEXTO LISTO PARA LLM:")
        print("="*80)
        print(llm_text[:1000] + "..." if len(llm_text) > 1000 else llm_text)
        print("\n" + "="*80)
        print(f"📝 Longitud total: {len(llm_text)} caracteres")

    else:
        print("Uso: python json_to_text.py <carpeta_con_jsons>")
        print("\nEjemplo: python json_to_text.py ./extracted")
        print("\nDevuelve un string listo para enviar al LLM")