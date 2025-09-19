"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { Camera, ImageUp, Loader2, RefreshCcw, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { API_URL, PREDICT_PATH } from "@/config";

type EmotionScore = {
  label: string;
  score: number; // 0..1
};

type PredictResponse = {
  emotions: EmotionScore[];
  top_emotion?: string;
};

export default function Home() {
  const [mode, setMode] = useState<"camera" | "upload">("upload");
  const [streaming, setStreaming] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const endpoint = useMemo(() => `${API_URL}${PREDICT_PATH}`, []);

  useEffect(() => {
    if (mode !== "camera") return;
    let stream: MediaStream | null = null;
    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setStreaming(true);
        }
      } catch (e) {
        setError("Camera access was denied or unavailable.");
      }
    };
    start();
    return () => {
      setStreaming(false);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [mode]);

  const resetAll = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  }, []);

  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);
    return await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", 0.9));
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const analyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let blob: Blob | null = null;
      if (mode === "camera") {
        blob = await captureFrame();
        if (!blob) throw new Error("Unable to capture image from camera");
      } else if (mode === "upload") {
        if (!file) throw new Error("Please choose an image first");
        blob = file;
      }

      const formData = new FormData();
      formData.append("file", blob as Blob, "frame.jpg");

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }

      // Robust JSON handling (even if content-type is wrong or body is stringified)
      let dataRaw: any;
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        dataRaw = await res.json();
      } else {
        const text = await res.text();
        try {
          dataRaw = JSON.parse(text);
        } catch {
          throw new Error(text || "Invalid JSON response");
        }
      }

      // Normalize into PredictResponse.emotions
      let emotions: EmotionScore[] | null = null;
      if (Array.isArray(dataRaw)) {
        emotions = dataRaw as EmotionScore[];
      } else if (dataRaw && Array.isArray(dataRaw.emotions)) {
        emotions = dataRaw.emotions as EmotionScore[];
      } else if (dataRaw?.result && Array.isArray(dataRaw.result.emotions)) {
        emotions = dataRaw.result.emotions as EmotionScore[];
      } else if (Array.isArray(dataRaw?.predictions)) {
        emotions = dataRaw.predictions as EmotionScore[];
      } else if (dataRaw && typeof dataRaw === "object" && ("class" in dataRaw) && ("confidence" in dataRaw)) {
        const label = String((dataRaw as any).class ?? (dataRaw as any).label ?? "unknown");
        const score = Number((dataRaw as any).confidence ?? (dataRaw as any).score);
        if (label && Number.isFinite(score)) {
          emotions = [{ label, score }];
        }
      } else if (dataRaw && typeof dataRaw === "object") {
        const entries = Object.entries(dataRaw).filter(([, v]) => typeof v === "number");
        if (entries.length) {
          emotions = entries.map(([label, score]) => ({ label, score: Number(score) }));
        }
      }

      if (!emotions || emotions.length === 0) {
        throw new Error("No emotions found in response");
      }

      const normalized = emotions
        .map((e) => ({ ...e, score: typeof e.score === "number" ? e.score : Number((e as any).score) }))
        .filter((e) => Number.isFinite(e.score))
        .sort((a, b) => b.score - a.score);

      setResult({ emotions: normalized, top_emotion: normalized[0]?.label });
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, [captureFrame, endpoint, file, mode]);

  const top = result?.emotions?.[0];

  return (
    <main className="min-h-screen p-6 sm:p-10">
      <section className="mx-auto max-w-6xl">
        <div className="mb-8 text-center">
          <Badge className="rounded-full px-3 py-1">Emotion AI</Badge>
          <h1 className="mt-4 text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight">
            Real‑time Facial Emotion Detection
          </h1>
          <p className="mt-3 text-muted-foreground max-w-2xl mx-auto">
            Use your camera or upload a photo. We'll detect emotions and show confidence scores.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6 items-start">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>Choose a source and capture or upload an image.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-end">
                <div className="w-full sm:w-56">
                  <Label htmlFor="source">Source</Label>
                  <Select value={mode} onValueChange={(v) => { setMode(v as any); resetAll(); }}>
                    <SelectTrigger id="source" className="mt-1">
                      <SelectValue placeholder="Select source" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Input Source</SelectLabel>
                        <SelectItem value="upload">upload</SelectItem>
                        <SelectItem value="camera">camera</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>

                {mode === "upload" ? (
                  <div className="flex-1">
                    <Label htmlFor="file">Image file</Label>
                    <Input id="file" type="file" accept="image/*" className="mt-1" onChange={onFileChange} />
                  </div>
                ) : (
                  <div className="flex-1" />
                )}

                <div className="flex gap-2">
                  <Button variant="secondary" onClick={resetAll} title="Reset">
                    <RefreshCcw className="size-4" />
                  </Button>
                  <Button onClick={analyze} disabled={loading || (mode === "upload" && !file)}>
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 size-4 animate-spin" /> Analyzing
                      </>
                    ) : (
                      <>
                        <Send className="mr-2 size-4" /> Analyze
                      </>
                    )}
                  </Button>
                </div>
              </div>

              <div className="mt-6 grid md:grid-cols-2 gap-4">
                <div className="relative aspect-video w-full overflow-hidden rounded-xl border">
                  {mode === "camera" ? (
                    <>
                      <video ref={videoRef} muted playsInline className="h-full w-full object-cover" />
                      {!streaming && (
                        <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
                          <Camera className="mr-2 size-4" /> Waiting for camera
                        </div>
                      )}
                    </>
                  ) : previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={previewUrl} alt="preview" className="h-full w-full object-cover" />
                  ) : (
                    <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
                      <ImageUp className="mr-2 size-4" /> Choose an image to preview
                    </div>
                  )}
                  <canvas ref={canvasRef} className="hidden" />
                </div>

                <div className="rounded-xl border p-4 bg-gradient-to-br from-purple-500/10 via-sky-500/10 to-emerald-500/10">
                  <h3 className="font-medium">Detected Emotions</h3>

                  {error && (
                    <div className="mt-3 text-sm text-red-600 dark:text-red-400">{error}</div>
                  )}

                  {!error && !result && (
                    <div className="mt-6 text-sm text-muted-foreground">
                      Run an analysis to see results here.
                    </div>
                  )}

                  {result && (
                    <div className="mt-4 space-y-3">
                      {top && (
                        <div className="mb-2">
                          <div className="text-sm text-muted-foreground">Top Emotion</div>
                          <div className="text-2xl font-semibold">{top.label} <span className="text-base font-normal text-muted-foreground">{`(${(top.score * 100).toFixed(1)}%)`}</span></div>
                        </div>
                      )}
                      {result.emotions.map((e) => (
                        <div key={e.label}>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="capitalize">{e.label}</span>
                            <span className="tabular-nums">{(e.score * 100).toFixed(1)}%</span>
                          </div>
                          <Progress value={Math.min(100, Math.max(0, e.score * 100))} />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden glass">
            <CardHeader>
              <CardTitle>How it looks</CardTitle>
              <CardDescription>Example output and guidance</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="relative w-full aspect-video overflow-hidden rounded-xl">
                <Image
                  src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1400&auto=format&fit=crop"
                  alt="Smiling person portrait"
                  fill
                  className="object-cover"
                  priority
                />
              </div>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• Good lighting improves detection accuracy.</li>
                <li>• Face should be clearly visible and centered.</li>
                <li>• For best accuracy, upload a close-up where the face occupies most of the frame.</li>
                <li>• We never store your images; they are sent directly to the model endpoint.</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>
    </main>
  );
}