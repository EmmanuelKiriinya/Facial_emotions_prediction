import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Smile } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import UploadBox from './UploadBox';
import Loader from './Loader';
import ResultDisplay from './ResultDisplay';

interface EmotionResult {
  emotion: string;
  confidence: number;
}

const FacialEmotionDetector = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<EmotionResult | null>(null);
  const [error, setError] = useState<string>('');
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError('');
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
    
    toast({
      title: "Image uploaded",
      description: "Ready to analyze emotion!",
    });
  };

  const analyzeEmotion = async () => {
    if (!selectedFile) {
      toast({
        title: "No image selected",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('https://facial-emotions-prediction.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // ✅ Fix: handle both "class" and "emotion"
      setResult({
        emotion: data.emotion || data.class || data.predicted_emotion || 'unknown',
        confidence: data.confidence || data.score || 0,
      });

      toast({
        title: "Analysis complete!",
        description: `Detected emotion: ${data.emotion || data.class || data.predicted_emotion}`,
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze emotion';
      setError(errorMessage);
      toast({
        title: "Analysis failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-8 px-4" style={{ background: 'var(--gradient-bg)' }}>
      <div className="max-w-2xl mx-auto">
        <Card className="shadow-2xl border-0 bg-card/90 backdrop-blur-sm">
          <CardHeader className="text-center pb-6">
            <div className="flex justify-center mb-4">
              <div className="p-4 rounded-full" style={{ background: 'var(--gradient-primary)' }}>
                <Smile className="w-8 h-8 text-white" />
              </div>
            </div>
            <CardTitle className="text-3xl font-bold text-foreground">
              Facial Emotion Detector
            </CardTitle>
            <p className="text-muted-foreground mt-2">
              Upload an image to analyze facial emotions using AI
            </p>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <UploadBox 
              onFileSelect={handleFileSelect}
              disabled={isLoading}
            />
            
            {selectedFile && !isLoading && (
              <div className="flex justify-center">
                <Button 
                  onClick={analyzeEmotion}
                  size="lg"
                  className="px-8"
                >
                  <Smile className="w-4 h-4 mr-2" />
                  Analyze Emotion
                </Button>
              </div>
            )}
            
            {isLoading && <Loader />}
            
            <ResultDisplay 
              result={result}
              imagePreview={imagePreview}
              error={error}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default FacialEmotionDetector;

