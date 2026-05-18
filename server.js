// const express = require("express");
// const app = express();

// app.use(express.static(__dirname));

// app.get("/:room", (req, res) => {
//   res.sendFile(__dirname + "/index.html");
// });

// const PORT = process.env.PORT || 3000;

// const server = app.listen(PORT, () => {
//   console.log(`✅ Node server running on http://localhost:${PORT}`);
// });

// const io = require("socket.io")(server, {
//   cors: { origin: "*" }
// });

// const rooms = {};

// io.on("connection", socket => {

//   socket.on("join-room", ({ roomId, peerId, username }) => {
//     socket.join(roomId);

//     if (!rooms[roomId]) rooms[roomId] = [];

//     // Send existing users to the newcomer
//     rooms[roomId].forEach(user => {
//       socket.emit("user-connected", user);
//     });

//     const userObj = { peerId, username };
//     if (!rooms[roomId].find(u => u.peerId === peerId)) {
//       rooms[roomId].push(userObj);
//     }

//     socket.roomId = roomId;
//     socket.peerId = peerId;

//     socket.to(roomId).emit("user-connected", userObj);
//     emitParticipants(roomId);
//   });

//   // MESSAGE SYSTEM
//   // data = { peerId, text, tts }
//   // Broadcast to everyone else in the room
//   socket.on("send-message", data => {
//     socket.to(socket.roomId).emit("receive-message", data);
//   });

//   function leaveRoom() {
//     const roomId = socket.roomId;
//     const peerId = socket.peerId;

//     if (roomId && rooms[roomId]) {
//       rooms[roomId] = rooms[roomId].filter(u => u.peerId !== peerId);
//       if (rooms[roomId].length === 0) {
//         delete rooms[roomId];
//       }
//       socket.to(roomId).emit("user-disconnected", peerId);
//       emitParticipants(roomId);
//     }
//   }

//   socket.on("leave-room", leaveRoom);
//   socket.on("disconnect", leaveRoom);

//   function emitParticipants(roomId) {
//     const count = rooms[roomId] ? rooms[roomId].length : 0;
//     io.to(roomId).emit("participants-count", count);
//   }
// });

const express = require("express");
const app = express();
const path = require("path");

// ── Proxy AI server ──
const { createProxyMiddleware } = require("http-proxy-middleware");
app.use("/ai", createProxyMiddleware({
  target: "http://localhost:5001",
  changeOrigin: true,
  pathRewrite: { "^/ai": "" },
}));

// ── Fix favicon 404 error ──
app.get("/favicon.ico", (req, res) => res.status(204).end());

// ── Serve static files EXCEPT index.html at root ──
// We serve css/js/assets normally, but NOT the default index.html
app.use(express.static(__dirname, { index: false }));

// ── Root "/" → always redirect to login ──
app.get("/", (req, res) => {
  res.redirect(301, "/login.html");
});

// ── login.html, register.html → serve directly ──
app.get("/login.html", (req, res) => {
  res.sendFile(path.join(__dirname, "login.html"));
});
app.get("/register.html", (req, res) => {
  res.sendFile(path.join(__dirname, "register.html"));
});

// ── index.html direct access ──
app.get("/index.html", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ── Any other path = room name → serve meeting page ──
app.get("/:room", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  console.log(`✅ Server running → http://localhost:${PORT}`);
  console.log(`   Opening "/" always redirects to login.html first.`);
});

const io = require("socket.io")(server, { cors: { origin: "*" } });
const rooms = {};

io.on("connection", socket => {

  socket.on("join-room", ({ roomId, peerId, username }) => {
    socket.join(roomId);
    if (!rooms[roomId]) rooms[roomId] = [];
    rooms[roomId].forEach(user => socket.emit("user-connected", user));
    if (!rooms[roomId].find(u => u.peerId === peerId)) {
      rooms[roomId].push({ peerId, username });
    }
    socket.roomId = roomId;
    socket.peerId = peerId;
    socket.to(roomId).emit("user-connected", { peerId, username });
    emitCount(roomId);
  });

  socket.on("send-message", data => {
    socket.to(socket.roomId).emit("receive-message", data);
  });

  function leave() {
    const { roomId, peerId } = socket;
    if (roomId && rooms[roomId]) {
      rooms[roomId] = rooms[roomId].filter(u => u.peerId !== peerId);
      socket.to(roomId).emit("user-disconnected", peerId);
      emitCount(roomId);
    }
  }
  socket.on("leave-room", leave);
  socket.on("disconnect", leave);

  function emitCount(roomId) {
    io.to(roomId).emit("participants-count",
      rooms[roomId] ? rooms[roomId].length : 0);
  }
});