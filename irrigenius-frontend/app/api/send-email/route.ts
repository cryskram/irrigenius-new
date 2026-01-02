import nodemailer from "nodemailer";

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const { mode, decision, temp, humidity, moisture, confidence } = body;

    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS,
      },
    });

    const message = `
Irrigenius Alert 🚨  

A STOP action was triggered.

Mode: ${mode}
Decision: ${decision}
Temperature: ${temp}
Humidity: ${humidity}
Moisture: ${moisture}
Confidence: ${confidence}

Please check the irrigation system immediately.
`;

    await transporter.sendMail({
      from: `"Irrigenius Alerts" <${process.env.EMAIL_USER}>`,
      to: process.env.ALERT_RECEIVER,
      subject: "🚨 Irrigenius Alert — WATER STOPPED",
      text: message,
    });

    return Response.json({ success: true });
  } catch (error) {
    console.error("Email error:", error);
    return Response.json({ success: false, error });
  }
}
