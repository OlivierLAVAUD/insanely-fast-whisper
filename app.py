import torch
from transformers import pipeline
from pyannote.audio import Pipeline
import torchaudio
import tempfile
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = ""

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: Optional[float] = None

@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    model_used: str
    diarization_applied: bool

class AudioTranscriber:
    def __init__(self, hf_token: str = HF_TOKEN):
        self.hf_token = hf_token
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.whisper_pipeline = None
        self.diarization_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize les modèles de manière lazy"""
        logger.info(f"Initializing models on device: {self.device}")
        
        # Whisper model
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device=self.device,
            token=self.hf_token
        )
    
    def _load_diarization_model(self):
        """Charge le modèle de diarization si nécessaire"""
        if self.diarization_pipeline is None:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                self.diarization_pipeline.to(torch.device(self.device))
                logger.info("Diarization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load diarization model: {e}")
                raise
    
    def resample_audio(self, audio_file_path: str, target_sr: int = 16000) -> str:
        """Resample l'audio et retourne le chemin du fichier temporaire"""
        try:
            waveform, original_sr = torchaudio.load(audio_file_path)
            
            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(original_sr, target_sr)
                waveform = resampler(waveform)
            
            # Crée un fichier temporaire pour l'audio resamplé
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(temp_file.name, waveform, target_sr)
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Audio resampling failed: {e}")
            raise
    
    def transcribe_audio(self, audio_file_path: str, language: str = "fr") -> List[Dict]:
        """Transcription de base avec Whisper"""
        try:
            outputs = self.whisper_pipeline(
                audio_file_path,
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=True,
                language=language
            )
            return outputs["chunks"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def apply_diarization(self, audio_file_path: str, segments: List[Dict]) -> List[TranscriptionSegment]:
        """Applique la diarization aux segments transcrits"""
        try:
            self._load_diarization_model()
            
            # Utilise l'audio resamplé pour la diarization
            resampled_audio_path = self.resample_audio(audio_file_path)
            
            try:
                diarization_result = self.diarization_pipeline(resampled_audio_path)
                
                # Map des speakers aux segments
                transcription_segments = []
                for segment in segments:
                    speaker = "SPEAKER_00"
                    for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                        if turn.start <= segment["timestamp"][0] <= turn.end:
                            speaker = speaker_label
                            break
                    
                    transcription_segments.append(TranscriptionSegment(
                        start=segment["timestamp"][0],
                        end=segment["timestamp"][1],
                        speaker=speaker,
                        text=segment["text"].strip(),
                        confidence=segment.get("confidence", 0.9)
                    ))
                
                return transcription_segments
                
            finally:
                # Nettoie le fichier temporaire
                if os.path.exists(resampled_audio_path):
                    os.unlink(resampled_audio_path)
                    
        except Exception as e:
            logger.warning(f"Diarization failed, returning segments without speaker identification: {e}")
            # Fallback: retourne les segments sans diarization
            return [
                TranscriptionSegment(
                    start=seg["timestamp"][0],
                    end=seg["timestamp"][1],
                    speaker="SPEAKER_00",
                    text=seg["text"].strip(),
                    confidence=seg.get("confidence", 0.9)
                )
                for seg in segments
            ]
    
    def process_audio(self, audio_file_path: str, enable_diarization: bool = True, 
                     language: str = "en") -> TranscriptionResult:
        """Processus complet de transcription"""
        start_time = datetime.now()
        
        try:
            # Transcription de base
            logger.info("Starting transcription...")
            raw_segments = self.transcribe_audio(audio_file_path, language)
            
            # Application de la diarization si demandée
            if enable_diarization:
                logger.info("Applying diarization...")
                segments = self.apply_diarization(audio_file_path, raw_segments)
            else:
                segments = [
                    TranscriptionSegment(
                        start=seg["timestamp"][0],
                        end=seg["timestamp"][1],
                        speaker="SPEAKER_00",
                        text=seg["text"].strip(),
                        confidence=seg.get("confidence", 0.9)
                    )
                    for seg in raw_segments
                ]
            
            # Calcul de la durée totale
            duration = segments[-1].end if segments else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TranscriptionResult(
                segments=segments,
                language=language,
                duration=duration,
                processing_time=processing_time,
                model_used="whisper-large-v3",
                diarization_applied=enable_diarization
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

class TranscriptionExporter:
    """Classe pour exporter les résultats dans différents formats"""
    
    @staticmethod
    def to_txt(result: TranscriptionResult, output_path: str) -> None:
        """Export en format texte"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription Results\n")
            f.write(f"====================\n")
            f.write(f"Language: {result.language}\n")
            f.write(f"Duration: {result.duration:.2f}s\n")
            f.write(f"Processing time: {result.processing_time:.2f}s\n")
            f.write(f"Diarization: {'Enabled' if result.diarization_applied else 'Disabled'}\n\n")
            
            for segment in result.segments:
                f.write(f"[{segment.speaker}] {segment.start:.1f}-{segment.end:.1f}: {segment.text}\n")
    
    @staticmethod
    def to_json(result: TranscriptionResult, output_path: str) -> None:
        """Export en format JSON"""
        output_data = {
            "metadata": {
                "language": result.language,
                "duration": result.duration,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "diarization_applied": result.diarization_applied,
                "generated_at": datetime.now().isoformat()
            },
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "confidence": seg.confidence
                }
                for seg in result.segments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def to_srt(result: TranscriptionResult, output_path: str) -> None:
        """Export en format SRT (sous-titres)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.segments, 1):
                # Format SRT timing
                start_time = TranscriptionExporter._format_srt_time(segment.start)
                end_time = TranscriptionExporter._format_srt_time(segment.end)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{segment.speaker}] {segment.text}\n\n")
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Convertit des secondes en format SRT HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

# Usage example
def main():
    audio_file = "audio.mp3"
    
    try:
        # Initialise le transcripteur
        transcriber = AudioTranscriber(HF_TOKEN)
        
        # Processus de transcription
        result = transcriber.process_audio(
            audio_file_path=audio_file,
            enable_diarization=True,
            language="en"
        )
        
        # Affiche les résultats
        print(f"\n=== TRANSCRIPTION COMPLÉTÉE ===")
        print(f"Durée audio: {result.duration:.2f}s")
        print(f"Temps de traitement: {result.processing_time:.2f}s")
        print(f"Segments: {len(result.segments)}")
        print(f"Diarization: {'Oui' if result.diarization_applied else 'Non'}")
        
        # Exporte les résultats
        exporter = TranscriptionExporter()
        exporter.to_txt(result, "transcription.txt")
        exporter.to_json(result, "transcription.json")
        exporter.to_srt(result, "transcription.srt")
        
        print(f"\nRésultats exportés dans:")
        print(f"- transcription.txt")
        print(f"- transcription.json") 
        print(f"- transcription.srt")
        
        # Affiche un aperçu
        print(f"\n=== APERÇU ===")
        for i, segment in enumerate(result.segments[:5]):  # Premier 5 segments
            print(f"[{segment.speaker}] {segment.start:.1f}-{segment.end:.1f}: {segment.text}")
        
        if len(result.segments) > 5:
            print(f"... et {len(result.segments) - 5} segments supplémentaires")
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        print("Échec du traitement. Vérifiez le fichier audio et la connexion.")

if __name__ == "__main__":
    main()