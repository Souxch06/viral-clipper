import type { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { id } = req.query;
  const store = (globalThis as any).__VC_JOBS || {};
  const job = store[id as string] || null;
  if (!job) return res.status(200).json({ status: 'unknown' });
  return res.status(200).json(job);
}
