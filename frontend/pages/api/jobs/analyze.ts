import type { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: 'Missing url' });

  // Ici on créerait un job réel (POST vers FastAPI), pour le demo on simule
  const jobId = 'job_' + Math.random().toString(36).slice(2, 9);
  // Simulated job state stored in-memory (for demo only)
  // In real app: call backend FastAPI to enqueue job and return job id
  globalThis.__VC_JOBS = globalThis.__VC_JOBS || {};
  (globalThis.__VC_JOBS as any)[jobId] = {
    status: 'processing',
    step: 'Téléchargement',
    clips: null
  };

  // Simulate completion after 6s (demo)
  setTimeout(() => {
    (globalThis.__VC_JOBS as any)[jobId] = {
      status: 'done',
      step: 'Rendering',
      clips: [
        { id: 'c1', score: 96, preview_url: '/sample/clip1.mp4', download_url: '/sample/clip1.mp4' },
        { id: 'c2', score: 91, preview_url: '/sample/clip2.mp4', download_url: '/sample/clip2.mp4' }
      ]
    };
  }, 6000);

  res.status(200).json({ job_id: jobId });
}
