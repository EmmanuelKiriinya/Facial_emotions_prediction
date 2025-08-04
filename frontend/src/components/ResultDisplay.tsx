import { CheckCircle, AlertCircle } from 'lucide-react';

interface EmotionResult {
  emotion: string;
  confidence: number;
}

interface ResultDisplayProps {
  result: EmotionResult | null;
  imagePreview: string | null;
  error?: string;
}

const ResultDisplay = ({ result, imagePreview, error }: ResultDisplayProps) => {
  if (error) {
    return (
      <div className="flex items-center gap-3 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
        <AlertCircle className="w-5 h-5 text-destructive" />
        <div>
          <p className="font-medium text-destructive">Error</p>
          <p className="text-sm text-destructive/80">{error}</p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="space-y-6">
      {imagePreview && (
        <div className="flex justify-center">
          <img
            src={imagePreview}
            alt="Uploaded image"
            className="max-w-full max-h-48 rounded-lg shadow-md object-contain"
          />
        </div>
      )}
      
      <div className="flex items-center gap-3 p-4 bg-success/10 border border-success/20 rounded-lg">
        <CheckCircle className="w-5 h-5 text-success" />
        <div className="flex-1">
          <p className="font-medium text-success">Analysis Complete</p>
          <div className="mt-2 space-y-1">
            <p className="text-lg font-semibold text-foreground">
              Emotion: <span className="text-primary capitalize">{result.emotion}</span>
            </p>
            <p className="text-sm text-muted-foreground">
              Confidence: <span className="font-medium">{(result.confidence * 100).toFixed(1)}%</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;