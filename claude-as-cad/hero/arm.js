// claude-as-cad hero — arm loader, forward kinematics and cursor-follow IK.
//
// Loads the static bundle written by tools/export_hero_arm.py (arm.json + arm.bin), rebuilds the
// URDF joint chain as a three.js scene graph (world frame is Z-up, metres), and provides a small
// damped-least-squares IK that drives the gripper TCP toward a target point.
import * as THREE from "./vendor/three.module.min.js";

const IK_JOINTS = ["J1", "J2", "J3", "J4", "J5"];

export async function loadArm(baseUrl) {
  const { manifest, bin } = await loadBundle(baseUrl, "arm");
  return buildArm(manifest, bin);
}

/**
 * Fetch assets/<name>.json + assets/<name>.bin. The manifest names the content hash of the .bin, so
 * a re-exported .bin always gets a new URL — a long-cached old .bin can never be paired with a new
 * manifest (offsets would run off its end).
 */
export async function loadBundle(baseUrl, name) {
  const manifest = await fetch(new URL(`assets/${name}.json`, baseUrl), { cache: "no-cache" }).then((r) => {
    if (!r.ok) throw new Error(`${name}.json ${r.status}`);
    return r.json();
  });
  const binUrl = new URL(`assets/${name}.bin?v=${manifest.binHash || "0"}`, baseUrl);
  const fetchBin = (cache) => fetch(binUrl, { cache }).then((r) => {
    if (!r.ok) throw new Error(`${name}.bin ${r.status}`);
    return r.arrayBuffer();
  });
  let bin = await fetchBin("default");
  if (manifest.binBytes && bin.byteLength !== manifest.binBytes) bin = await fetchBin("reload");
  if (manifest.binBytes && bin.byteLength !== manifest.binBytes) {
    throw new Error(`${name}.bin is ${bin.byteLength} bytes, manifest expects ${manifest.binBytes}`);
  }
  return { manifest, bin };
}

export function buildGeometry(entry, bin) {
  const [pOff, pLen] = entry.positions;
  const [nOff, nLen] = entry.normals;
  const [iOff, iLen] = entry.indices;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bin, pOff, pLen / 4), 3));
  const normals = new THREE.InterleavedBuffer(new Int8Array(bin, nOff, nLen), 4);
  geometry.setAttribute("normal", new THREE.InterleavedBufferAttribute(normals, 3, 0, true));
  const indices = entry.indexType === "u16" ? new Uint16Array(bin, iOff, iLen / 2) : new Uint32Array(bin, iOff, iLen / 4);
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeBoundingSphere();
  return geometry;
}

function materialFor(entry) {
  if (entry.role === "highlight") {
    return new THREE.MeshBasicMaterial({
      color: entry.color, transparent: true, opacity: 0, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
    });
  }
  const color = new THREE.Color(entry.color);
  const servo = entry.role === "servo";
  return new THREE.MeshStandardMaterial({
    color,
    roughness: servo ? 0.62 : 0.5,
    metalness: entry.role === "connector" || entry.role === "yoke" ? 0.18 : 0.04
  });
}

function buildArm(manifest, bin) {
  const root = new THREE.Group();
  root.name = "world";
  const links = new Map([[manifest.rootLink, root]]);
  const joints = new Map();

  // URDF joint chain: parent link -> origin (xyz + rpy) -> motion (rotation about axis) -> child link.
  const pending = [...manifest.joints];
  while (pending.length) {
    const index = pending.findIndex((joint) => links.has(joint.parent));
    if (index < 0) break;
    const joint = pending.splice(index, 1)[0];
    const origin = new THREE.Group();
    origin.position.fromArray(joint.xyz);
    origin.rotation.set(joint.rpy[0], joint.rpy[1], joint.rpy[2], "ZYX");
    const motion = new THREE.Group();
    motion.name = joint.child;
    origin.add(motion);
    links.get(joint.parent).add(origin);
    links.set(joint.child, motion);
    if (joint.type === "revolute" || joint.type === "continuous") {
      joints.set(joint.name, {
        name: joint.name,
        axis: new THREE.Vector3().fromArray(joint.axis).normalize(),
        limit: joint.limit || [-Math.PI, Math.PI],
        group: motion,
        value: 0
      });
    }
  }

  const meshes = new Map();
  for (const entry of manifest.meshes) {
    const mesh = new THREE.Mesh(buildGeometry(entry, bin), materialFor(entry));
    mesh.name = entry.name;
    mesh.userData.role = entry.role;
    mesh.userData.link = entry.link;
    if (entry.triMap) {
      // web triangle -> original STL triangle, and the STL -> link transform (click-to-reference)
      const [mOff, mLen] = entry.triMap;
      mesh.userData.triMap = entry.triMapType === "u16" ? new Uint16Array(bin, mOff, mLen / 2) : new Uint32Array(bin, mOff, mLen / 4);
      const linkFromStl = new THREE.Matrix4().fromArray(entry.visualMatrix).transpose();
      mesh.userData.stlFromLink = linkFromStl.clone().invert();
      // metres per STL unit (URDF mesh scale) — STLs are mostly authored in mm (scale 0.001)
      mesh.userData.stlUnit = new THREE.Vector3().setFromMatrixColumn(linkFromStl, 0).length();
    }
    if (entry.role === "highlight") {
      mesh.renderOrder = 2;
      mesh.visible = false;
    } else {
      mesh.castShadow = true;
      mesh.receiveShadow = false;
    }
    links.get(entry.link)?.add(mesh);
    meshes.set(entry.name, mesh);
  }

  const tcpLocal = new THREE.Vector3().fromArray(manifest.tcp.point);
  const tcpLink = links.get(manifest.tcp.link);
  const tmp = new THREE.Vector3();

  const arm = {
    root,
    links,
    joints,
    meshes,
    manifest,
    targets: manifest.targets,
    setJoint(name, value) {
      const joint = joints.get(name);
      if (!joint) return;
      const clamped = Math.min(joint.limit[1], Math.max(joint.limit[0], value));
      joint.value = clamped;
      joint.group.quaternion.setFromAxisAngle(joint.axis, clamped);
    },
    getPose() {
      const pose = {};
      for (const [name, joint] of joints) pose[name] = joint.value;
      return pose;
    },
    setPose(pose) {
      for (const [name, value] of Object.entries(pose)) arm.setJoint(name, value);
      root.updateMatrixWorld(true);
    },
    tcpWorld(out = new THREE.Vector3()) {
      return tcpLink.localToWorld(out.copy(tcpLocal));
    },
    linkWorldPoint(linkName, local, out = new THREE.Vector3()) {
      return links.get(linkName).localToWorld(out.copy(local));
    },
    /** Joint position in world (pivot of the motion group). */
    jointWorld(name, out = new THREE.Vector3()) {
      return joints.get(name).group.getWorldPosition(out);
    },
    /**
     * A few damped-least-squares iterations moving the TCP toward `target`.
     * Position-only over J1..J5, with a light pull toward `rest` so the elbow/wrist stay natural.
     * `maxStep` (rad) rate-limits each joint so motion stays smooth.
     */
    solveIK(target, rest, { iterations = 3, lambda = 0.045, maxStep = 0.05, restPull = 0.02 } = {}) {
      const axisW = new THREE.Vector3();
      const pivot = new THREE.Vector3();
      const col = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      for (let iter = 0; iter < iterations; iter += 1) {
        root.updateMatrixWorld(true);
        const tcp = arm.tcpWorld(tmp);
        const error = new THREE.Vector3().subVectors(target, tcp);
        const dist = error.length();
        if (dist < 0.0004) return dist;
        if (dist > 0.04) error.multiplyScalar(0.04 / dist);
        // Jacobian columns: axis_world x (tcp - pivot)
        const cols = IK_JOINTS.map((name) => {
          const joint = joints.get(name);
          joint.group.getWorldQuaternion(quat);
          axisW.copy(joint.axis).applyQuaternion(quat);
          joint.group.getWorldPosition(pivot);
          return col.subVectors(tcp, pivot).clone().crossVectors(axisW, new THREE.Vector3().subVectors(tcp, pivot));
        });
        // dq = J^T (J J^T + λ² I)^-1 e
        const JJt = new THREE.Matrix3().set(0, 0, 0, 0, 0, 0, 0, 0, 0);
        const m = JJt.elements; // column-major
        for (const c of cols) {
          const a = [c.x, c.y, c.z];
          for (let r = 0; r < 3; r += 1) {
            for (let k = 0; k < 3; k += 1) m[k * 3 + r] += a[r] * a[k];
          }
        }
        for (let d = 0; d < 3; d += 1) m[d * 3 + d] += lambda * lambda;
        const y = error.clone().applyMatrix3(JJt.invert());
        IK_JOINTS.forEach((name, i) => {
          const joint = joints.get(name);
          let dq = cols[i].dot(y);
          if (rest && rest[name] !== undefined) dq += restPull * (rest[name] - joint.value);
          dq = Math.max(-maxStep, Math.min(maxStep, dq));
          arm.setJoint(name, joint.value + dq);
        });
      }
      root.updateMatrixWorld(true);
      return arm.tcpWorld(tmp).distanceTo(target);
    }
  };
  root.updateMatrixWorld(true);
  return arm;
}
