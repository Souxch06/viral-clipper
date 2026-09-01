import type { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';

export const config = {
  api: {
    bodyParser: false
  }
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();
  const form = formidable({ multiples: false });
  form.parse(req as any, (err, fields) => {
    if (err) {
      console.error(err);
      return res.status(500).end();
    }
    const url = fields.url || fields.text || fields.title;
    const redirectTo = '/?shared=' + encodeURIComponent(String(url || ''));
    res.writeHead(303, { Location: redirectTo });
    res.end();
  });
}
